import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import math

@dataclass
class ModelArgs:
    max_batch_size: int = 32
    max_seq_len: int = 512
    vocab_size: int = 50257
    hidden_dim: int = 768
    input_dim: int = 1536
    moe_inter_dim: int = 512
    n_layers: int = 6
    num_heads: int = 12
    device = 'cpu'
    # moe
    num_experts: int = 4
    num_share_experts: int = 1
    topk: int = 1
    router_scale: float = 1.
    # mla
    q_lora_rank: int = 128
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # mtp
    mtp_depth: int = 1
    mtp_lambda: float = 0.3

def precompute_theta_pos_frequencies(head_dim: int, max_seq_len: int, device: str, theta: float = 10000.0):
    # according to the paper, head_dim should be even
    assert head_dim % 2 == 0, "Head dim should be even"
    # Shape: (head_dim / 2), according to the paper, represents theta [0, 2, ..., d - 2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim / 2), formula: theta_i = 10000 ^ (-2(i - 1) / dim), for i = 1, 2, ..., d / 2
    theta = 1 / (theta ** (theta_numerator / head_dim)).to(device)
    # Shape: (max_seq_len), according to the paper, represents the position [1, 2, ..., n]
    m = torch.arange(0, max_seq_len, device=device)
    # Shape: (max_seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # Shape: (max_seq_len, head_dim / 2)
    # the first numberrepresents the cos, the second numer represents the sin
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    # (b, s, h, d) -> (b, s, h, d / 2, 2) -> (b, s, h, d / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (b, s, h, d / 2) * (1, s, 1, d / 2) -> (b, s, h, d / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_complex = x_complex * freqs_complex
    # (b, s, h, d / 2) -> (b, s, h, d / 2, 2) -> (b, s, h, d)
    x_out = torch.view_as_real(x_complex).reshape(*x.shape)
    return x_out.type_as(x)

class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_down_proj = nn.Linear(self.hidden_dim, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim,  bias=False)

        self.kv_down_proj = nn.Linear(self.hidden_dim, self.qk_rope_head_dim + self.kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.v_head_dim + self.qk_nope_head_dim), bias=False)

        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_dim)

        self.register_buffer("k_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.num_heads, self.qk_head_dim), persistent=False)
        self.register_buffer("v_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.num_heads, self.v_head_dim), persistent=False)

    def forward(self, x, freq_complex, start_pos):
        batch_size, seq_len, _ = x.size()
        end_pos = start_pos + seq_len

        q = self.q_up_proj(self.q_norm(self.q_down_proj(x)))
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = apply_rotary_embeddings(q_rope, freq_complex)

        kv = self.kv_down_proj(x)
        k_rope, kv = torch.split(kv, [self.qk_rope_head_dim, self.kv_lora_rank], dim=-1)
        k_rope = apply_rotary_embeddings(k_rope.unsqueeze(2), freq_complex)
        
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = self.kv_up_proj(self.kv_norm(kv))
        kv = kv.view(batch_size, seq_len, self.num_heads, self.v_head_dim + self.qk_nope_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(-1, -1, self.num_heads, -1)], dim=-1)

        self.k_cache[:batch_size, start_pos:end_pos] = k
        self.v_cache[:batch_size, start_pos:end_pos] = v

        k = self.k_cache[:batch_size, :end_pos]
        v = self.v_cache[:batch_size, :end_pos]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        score = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        score = score.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        out = self.out_proj(score)

        return out

class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.topk = config.topk
        self.router_scale = config.router_scale
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim))
        self.bias = nn.Parameter(torch.empty(config.num_experts), requires_grad=False)
        self.u = 0.0001
    
    def forward(self, x):
        scores = F.linear(x, self.weight)
        scores = scores.softmax(dim=-1)
        original_scores = scores
        # load balancing
        scores = scores + self.bias

        selected_experts = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(dim=-1, index=selected_experts)

        weights *= self.router_scale
        return weights.type_as(x), selected_experts
    
class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, input_dim)
        self.w3 = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))

class SharedSparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts

        self.topk = config.topk
        self.gate = Gate(config)
        self.experts = nn.ModuleList([Expert(config.hidden_dim, config.moe_inter_dim) for _ in range(config.num_experts)])
        self.shared_experts = Expert(config.hidden_dim, config.moe_inter_dim)

        self.activations = np.zeros(config.num_experts)
    
    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.hidden_dim)
        weights, selected_experts = self.gate(x)
        y = torch.zeros_like(x)
        # a tensor records the number of usage of each expert
        counts = torch.bincount(selected_experts.flatten(), minlength=self.num_experts).to(x.device)
        self.activations += counts.cpu().numpy()
        # load balance
        self.gate.bias += self.gate.u * torch.sign(counts.float().mean() - counts)
        for i in range(self.num_experts):
            # this expert is not selected by any sample
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top_x = torch.where(selected_experts == i)
            # y[idx] = expert(x[idx]) * weights[idx, top_x, None]
            y.index_add_(0, idx, expert(x[idx]) * weights[idx, top_x, None])
        z = self.shared_experts(x)
        return (y + z).view(shape)
    
    def get_expert_activations(self):
        print(self.activations)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_dim)
        self.ffn_norm = nn.RMSNorm(config.hidden_dim)
        self.attn = MLA(config)
        self.ffn = SharedSparseMoE(config)
    
    def forward(self, x, freq_complex, start_pos):
        x = x + self.attn(self.attn_norm(x), freq_complex, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class MTPModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.trm = Block(config)
        self.rms1 = nn.RMSNorm(config.hidden_dim)
        self.rms2 = nn.RMSNorm(config.hidden_dim)
    
    
    def forward(self, h_prev, embed_tokens, freq_complex, start_pos):
        combined = torch.cat([
            self.rms1(h_prev),
            self.rms2(embed_tokens)
        ], dim=2)
        h_prime = self.proj(combined)
        h_current = self.trm(h_prime, freq_complex, start_pos)
        return h_current

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(Block(config))
        self.mtp = nn.ModuleList([MTPModule(config) for _ in range(self.config.mtp_depth)])
        # self.mtp_lambda = nn.Parameter(torch.empty(1), requires_grad=True)
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

        self.freq_complex = precompute_theta_pos_frequencies(config.hidden_dim // config.num_heads, config.max_seq_len * 2, device=config.device)
    
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            for param in self.parameters():
                torch.nn.init.normal_(
                    param, 
                    mean=0.0,
                    std=math.sqrt(0.006)
                )

    def forward(self, tokens, start_pos=0, depth=None, targets=None):
        if depth is None:
            assert targets is None
            main_tokens = tokens
            seqlen = tokens.size(1)
        else:    
            main_tokens = tokens[:, :-depth]
            seqlen = main_tokens.size(1)

        h = self.embed(main_tokens)
        freq_complex = self.freq_complex[start_pos : start_pos + seqlen]

        for layer in self.layers:
            h = layer(h, freq_complex, start_pos)
        h_main = self.norm(h)
        logits_main = self.output(h_main)

        if targets is None:
            return logits_main
        
        h_prev = h_main
        loss_mtp = 0
        for i in range(self.config.mtp_depth):
            indices = slice(i + 1, seqlen + i + 1)
            freq_complex = self.freq_complex[indices]

            mtp_inputs = tokens[:, indices]
            mtp_targets = targets[:, indices]

            emb_inputs = self.embed(mtp_inputs)

            h_current = self.mtp[i](h_prev, emb_inputs, freq_complex)

            h_prev = h_current

            all_logits = self.output(h_current)

            mtp_loss = F.cross_entropy(
                all_logits.reshape(-1, self.config.vocab_size),
                mtp_targets.reshape(-1)
            )

            loss_mtp += mtp_loss/ self.config.mtp_depth
        
        loss_main = F.cross_entropy(logits_main.view(-1, logits_main.size(-1)), targets[:, :seqlen].reshape(-1))
        loss_mtp = loss_mtp * self.config.mtp_lambda
        loss_main += loss_mtp
            
        return logits_main, loss_main, loss_mtp.item()
    
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

DeepSeekConfig = ModelArgs()
DeepSeekConfig.device = device