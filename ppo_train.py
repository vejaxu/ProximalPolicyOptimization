from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset as datasets_Dataset


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        super().__init__()
        self.prompts = prompts
        self.tokenizer = tokenizer

        self.final_prompts = []
        for prompt in self.prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt
        
            self.final_prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.final_prompts[index]


def format_prompt(example, tokenizer, apply_chat_template=True):
    prompt = example('prompt')
    if apply_chat_template:
        content = [{"role": "user", "content": prompt}]
        example["formatted_prompt"] = tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
    else:
        example["formatted_prompt"] = tokenizer.bos_token + prompt
    return example


def create_prompt_dataset(prompts, tokenizer, apply_chat_template=False):
    dataset = datasets_Dataset.from_dict({"prompt": prompts})
    dataset = dataset.map(lambda x: format_prompt(x, tokenizer, apply_chat_template), 
                          remove_columns=["prompt"]
    )
    return dataset


class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1)

    
    def forward(self, input_ids, attention_mask, num_actions):
        hidden_states = self.base_model(input_ids, attention_mask).last_hidden_state # b n d
        value_model_output = self.value_head(hidden_states) # b n 1
        values = value_model_output.squeeze(-1)[:, - num_actions: ] # b n 1 -> b n -> b num_actions 我们只给生成部分进行赋值
        return values
    

def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = - torch.min(surr1, surr2)
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean() # 对每个样本的有效位置计算平均loss，再对每个样本取平均


def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps: float = None):
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2
        
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()
    

@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor


@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None


@dataclass
class BufferItem:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]


class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []

    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
            "seqs", 
            "action_log_probs", 
            "values", 
            "returns", 
            "advantages", 
            "attention_mask", 
            "action_mask", 
            "num_actions"
        )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
        
        self.buffer.extend(batch)
        if len(self.buffer) > self.limit:
            self.buffer = self.buffer[len(self.buffer) - self.limit: ]

    def get_batchs(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]


def collate_fn(batch):

    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []
    
    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)
    
    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask, action_mask.size(1))


def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    samples_list = []
    model.eval() # 采样样本时模型应该eval
    all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in prompts], []) # 一共包含n_samples_per_prompt * len(prompts)个样本
    for i in range(0, len(all_prompts), micro_rollout_batch_size): # micro_rollout_batch_size与n_samples_per_prompt相对应上，这样每次采样还是对同一个prompt进行采样
        prompts = all_prompts[i: i + micro_rollout_batch_size]
        inputs = actor_tokenizer(prompts, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        seqs = model.generate(**inputs.to(device), 
                            max_new_tokens = max_new_tokens, # 最大生成长度
                            eos_token_id = eos_token_id, 
                            pad_token_id = pad_token_id)
        
        if seqs.shape[1] >= max_new_tokens + max_length: # 生成长度 + prompt长度
            seqs = seqs[:, :max_new_tokens + max_length]
        else:
            seqs = torch.cat([seqs, torch.full((seqs.shape[0], max_new_tokens + max_length - seqs.shape[1]), fill_value=pad_token_id, device=seqs.device)], dim=1)

        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        # .ne是tensor自带的操作，逐个元素判断哪些元素不是pad_token_id
        ans = seqs[:, input_ids.shape[1]:] # 生成的部分
        action_mask = (ans.ne(eos_token_id) & ans.ne(pad_token_id)).to(dtype=torch.long) # 生成部分的mask，去掉eos_token_id和pad_token_id，获得真正生成部分的mask

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask, # 整个样本的mask，包含prompt与response
            action_mask=action_mask, # response部分的mask，包含eos_token_id和pad_token_id
            num_actions=action_mask.shape[1],
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
        samples_list.append(samples)
    print(f"samples_list length: {len(samples_list)}")

    return samples_list


def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):
    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask # 只计算生成部分的KL散度
    return log_ratio


def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):

        kl_divergence_estimate = -kl_ctl * kl
        rewards = kl_divergence_estimate # b * num_actions

        ends = action_mask.sum(1) + 1
        
        if not isinstance(clip_reward_value, torch.Tensor):
            clip_reward_value = torch.tensor(clip_reward_value).to(r.device)
    
        reward_clip = torch.clamp(r, -clip_reward_value,
                                  clip_reward_value)
        batch_size = r.size(0)
        for j in range(batch_size):
            rewards[j, :ends[j]][-1] += reward_clip[j, 0]

        return rewards


# A(t) = R(t) + gam * V(t+1) - V(t)
# gae:A(t) = R(t) + gam * V(t+1) - V(t) + gamc *lam *A(t+1)
# 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
# A(T-1) = R(T-1) + gam * V(T) - V(T-1) + gam * lam * A(T) 知道A(T)可计算A(T-1) 依次类推
# returns(t) = A(t) + V(t) = = R(t) + gam * (V(t+1) + lam * A(t+1))
def get_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float):
    
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)
    
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    for t in reversed(range(response_length)):
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns


def generate_experiences(samples_list):
    """
    samples = Samples(
        seqs=seqs,
        attention_mask=attention_mask,
        action_mask=action_mask, # 对于每个句子，找到其生成的部分，不包含pad部分
        num_actions=action_mask.size(1), # 包含pad部分
        packed_seq_lens=None,
        response_length=action_mask.float().sum(dim=-1),
        total_length=attention_mask.float().sum(dim=-1),
    )
    """
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []
    
    for samples in samples_list:
        seqs = samples.seqs
        print(f"seqs shape: {seqs.shape}")
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        print(f"action_mask shape: {action_mask.shape}")
        num_actions = samples.num_actions
        print(f"num_actions: {num_actions}")
        with torch.no_grad():
            # 计算策略模型输出token的概率
            output = actor_model(seqs, attention_mask=attention_mask) # b n 
            logits = output.logits # b n vocab_size
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1) # b (n-1) vocab_size
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1)) # 这是上一行操作的标签，后一个token是前一个的label
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:] # 取出生成部分的log概率值 b * num_actions
            #计算参考模型输出token的概率
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]
            # 计算价值
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)
            print(f"value shape: {value.shape}")
            # 转换成文本
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # 计算奖励模型的奖励值
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True) # prompt + response
            r = reward_model(**reward_model_inputs.to(device)).logits # b * 1 一个句子对应一个分数，因此，奖励模型也是很重要的一个组件
            print(f"reward shape: {r.shape}")
            # 计算kl散度
            kl = compute_approx_kl(
                    action_log_probs,
                    ref_action_log_probs,
                    action_mask=action_mask).to(device)
            print(f"kl shape: {kl.shape}")
            # 计算实际奖励
            rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2)
            print(f"rewards shape: {rewards.shape}")
            # 计算优势和回报
            advantages, returns = get_advantages_and_returns(value, rewards, action_mask, gamma=0.1, lambd=0.2)
        # actor_model.train()
        # critic_model.train()

        experiences.append(
            Experience(
                seqs,
                action_log_probs.detach(),
                value.detach(),
                returns.detach(),
                advantages.detach(),
                attention_mask,
                action_mask,
                r.detach(),
                samples.response_length,
                samples.total_length,
                num_actions,
                kl.detach(),
                )
            )
    print(f"experiences length: {len(experiences)}")

    return experiences
    

def train_step(experience, steps):
    
    actor_model.train()
    optimizer_actor.zero_grad()

    
    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns
    
    logits = actor_model(
            sequences,
            attention_mask=attention_mask).logits
    
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
  
    
    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs, advantages, action_mask=action_mask)
    policy_loss.backward()
    optimizer_actor.step()  
    writer.add_scalar("policy_loss", policy_loss.item(), steps)
    
    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")
    

def train():
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader: # 每次取8条prompt进行采样
            samples = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size) # 每次取8个prompt，每个prompt生成2个样本，并且在采样过程中，一次只采样2条数据
            # samples是一个列表，元素是Samples对象，长度为8，每个Sample包含了2条采样样本
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size, shuffle=True, collate_fn=collate_fn)
            torch.cuda.empty_cache()
            for _ in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
            
            buffer.clear()
        
            torch.cuda.empty_cache()
            

if __name__ == "__main__":
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    episodes = 3 # 一共迭代多少轮
    max_epochs = 5 # 生成一次经验，训练的轮数
    rollout_batch_size = 8 # 一次从提示词数据集中取多少条数据用于生成经验
    micro_rollout_batch_size = 2 # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    n_samples_per_prompt = 2 # 一个提示词生成多少个样本
    max_new_tokens = 50 # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_length = 256
    micro_train_batch_size = 2 # 实际训练的batch_size大小，一次取多少条数据用于更新参数

    writer = SummaryWriter('runs')

    actor_model = AutoModelForCausalLM.from_pretrained('/home/xwj/Model/qwen2.5-0.5b-instruct').to(device)
    ref_model = AutoModelForCausalLM.from_pretrained('/home/xwj/Model/qwen2.5-0.5b-instruct').to(device)
    reward_model = AutoModelForSequenceClassification.from_pretrained('/home/xwj/Model/reward-model-deberta-v3-large-v2').to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained('/home/xwj/Model/qwen2.5-0.5b-instruct')
    reward_tokenizer = AutoTokenizer.from_pretrained('/home/xwj/Model/reward-model-deberta-v3-large-v2')
    critic_model = Critic(actor_model.base_model).to(device)
    
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)
    
    actor_tokenizer.padding_side = 'left' # 对最后一个token打分才有意义
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id
    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=False)
    # formatted_dataset = create_prompt_dataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    # torch_dataset = formatted_dataset.with_format("torch")  # 返回一个 PyTorch 兼容的 dataset
    # prompts_dataloader = DataLoader(torch_dataset, batch_size=rollout_batch_size, shuffle=False)
   
    train()
    # episode * 8条经验 * 每个经验用5次 / 每次用2条经验 = 60