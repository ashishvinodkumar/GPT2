from dataclasses import dataclass
from IPython.utils import process
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import inspect
import os
from torch.distributed import init_process_group, destroy_process_group
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1,1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # Batch Size, Token/Sequence Length, Embedding Dimension
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        # att = (q @ k.transpose(-2, -1)) * (1/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # Sometimes also referred to as Feed Forward Network (FFN)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # This is where they communicate and become context rich.
        x = x + self.mlp(self.ln_2(x)) # This is where they think individually
        return x

@dataclass
class GPT2Config:
    block_size: int = 1024 # Max Sequence Length
    vocab_size: int = 50257 # Number of tokens. 50k Byte Pair Encoding merges
    n_layer: int = 12 # Number of layers
    n_head: int = 12 # Number of heads
    n_embd: int = 768 # Number of embedding dimensions.

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte =  nn.Embedding(config.vocab_size, config.n_embd), # wte: Weights of Token Embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # wpe: Weights of Posentional Embeddings. These are trainable parameters unlike the original Transformer architecture.
            h = nn.ModuleList([Block(config) for _ in range(0, config.n_layer)]), # h: Stands for hidden layers. Creates 'n' layer blocks.
            ln_f = nn.LayerNorm(config.n_embd), # ln_f: Layer Norm. Not in the original transformer architecture
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # lm_head: Final layer.

        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # To match GPT2 paper, we need to manually apply mean and std to random weight initializations
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # Scale down the standard deviation to keep it closer to 1.
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size() # idx is of shape (B, T). B=Batch. T=Token
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # Forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # Forward the final layer-norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, Vocab Size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    

class DataLoaderLite:
    def __init__(self, B, T, input_text, process_rank, num_processes):
        self.B = B
        self.T = T
        self.input_text = input_text
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open( self.input_text, 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'1 Epoch = {len(self.tokens)// (B*T)} batches')

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # Inputs
        y = (buf[1:]).view(B, T) # Targets

        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # If loading the next batch would be out of bounds, reset.
        if self.current_position + (B*T*self.num_processes+1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y




# -----------------------------------------
# Distributed Data Parallel
# -----------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1 # Is this ddp run?
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'Using Device: {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, ~0.5M, as per GPT paper.
B = 16 # Micro Batch Size
T = 1024 # Max Sequence Length
assert total_batch_size % (B * T * ddp_world_size) == 0, "Total Batch Size must be divisible by B*T*dd_world_size"
grand_accum_steps = total_batch_size // (B*T*ddp_world_size)

if master_process:
  print(f'Total Desired Batch Size: {total_batch_size}')
  print(f'Grand Accumulate Steps: {grand_accum_steps}')

print(f'I am GPU {ddp_rank=}')

input_text = './data/input.txt'
train_loader = DataLoaderLite(B=B, T=T, input_text=input_text, process_rank=ddp_rank, num_processes=ddp_world_size)

# Set precision to TF32 when available. Will speed up total performance.
# TF32 will reduce the decimal precision.
torch.set_float32_matmul_precision('high')

# Initialize model
model = GPT2(GPT2Config(vocab_size=50304)) # Initializing with random weights. Not using HF model.
model.to(device)
model = torch.compile(model)

# Wrap Model with Torch DDP.
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
# Cosine decay learning rate with warm-up.
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
  # Linear warmp for warm_iter steps
  if it < warmup_steps:
    return max_lr * (it+1) / warmup_steps
  if it > max_steps:
    return min_lr
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0.0 <= decay_ratio <= 1.0
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, betas=(0.9, 0.95), learning_rate=6e-4, device_type=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grand_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(x, y)
        loss = loss / grand_accum_steps
        loss_accum += loss.detach()

        # Only at the very last step, run all-reduce to synchronize weights.
        # Official way is to use no_sync context manager.
        if ddp:
          model.require_backward_grad_sync = (micro_step == grand_accum_steps-1)
        loss.backward()

    if ddp:
      dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize() # Wait for gpu to finish work.
    t1 = time.time()
    dt = round((t1 - t0)*1000, 3) # time difference in ms.
    tokens_per_second = round((train_loader.B * train_loader.T * grand_accum_steps * ddp_world_size) / (t1-t0), 3)
    
    if master_process:
      print(f'step: {step} | loss: {loss_accum.item():.4e} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt} | tokens/sec: {tokens_per_second}')

if ddp:
  destroy_process_group()
