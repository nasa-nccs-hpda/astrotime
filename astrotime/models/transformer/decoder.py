import torch, math
from torch import nn
from typing import List, Tuple
from omegaconf import DictConfig
import torch.nn.functional as F

class RMSNorm(torch.nn.Module):

  def __init__( self, cfg: DictConfig ):
    super().__init__()
    self.eps = 1e-6
    self.weight = nn.Parameter(torch.zeros(cfg.hidden_size))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float())
    output = output * self.weight.float()
    return output.type_as(x)


class TransformerMLP(nn.Module):

  def __init__( self, cfg: DictConfig ):
    super().__init__()
    self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
    self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
    self.layer_norm = nn.LayerNorm(normalized_shape=cfg.hidden_size, eps=1e-6)

  def forward(self, x ):
    gate_inp = self.layer_norm(x)
    gate = self.gate_proj(gate_inp)
    gate = F.relu(gate)
    outputs = self.down_proj(gate)
    return outputs + x


class TimesFMAttention(nn.Module):

  def __init__(self, cfg: DictConfig ):
    super().__init__()

    self.num_heads: int = cfg.num_heads
    self.num_kv_heads: int = cfg.num_kv_heads
    assert self.num_heads % self.num_kv_heads == 0
    self.num_queries_per_kv: int = self.num_heads // self.num_kv_heads
    self.hidden_size: int = cfg.hidden_size
    self.head_dim: int = cfg.head_dim

    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = nn.Parameter( torch.empty((self.head_dim,), dtype=torch.float32),)
    self.qkv_proj = nn.Linear( self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim )
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

  def _per_dim_scaling(self, query: torch.Tensor) -> torch.Tensor:
    # [batch_size, n_local_heads, input_len, head_dim]
    r_softplus_0 = 1.442695041
    softplus_func = torch.nn.Softplus()
    scale = r_softplus_0 / math.sqrt(self.head_dim)
    scale = scale * softplus_func(self.scaling)
    return query * scale[None, None, None, :]

  def forward( self, hidden_states: torch.Tensor ) -> torch.Tensor:
    hidden_states_shape = hidden_states.shape
    assert len(hidden_states_shape) == 3
    batch_size, input_len, _ = hidden_states_shape

    qkv = self.qkv_proj(hidden_states)
    xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
    xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
    xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)
    xq = self._per_dim_scaling(xq)

    if self.num_kv_heads != self.num_heads:
      # [batch_size, max_seq_len, n_local_heads, head_dim]
      xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
      xv = torch.repeat_interleave(xv, self.num_queries_per_kv, dim=2)

    # [batch_size, n_local_heads, input_len, head_dim]
    q = xq.transpose(1, 2)
    # [batch_size, n_local_heads, max_seq_len, head_dim]
    k = xk.transpose(1, 2)
    v = xv.transpose(1, 2)

    # [batch_size, n_local_heads, input_len, max_seq_len]
    scores = torch.matmul(q, k.transpose(2, 3))
    scores = F.softmax(scores.float(), dim=-1).type_as(q)

    # [batch_size, n_local_heads, input_len, head_dim]
    output = torch.matmul(scores, v)
    # return scores, output.transpose(1, 2).contiguous()

    # [batch_size, input_len, hidden_dim]
    output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
    output = self.o_proj(output)
    return output


class TimesFMDecoderLayer(nn.Module):

  def __init__(self, cfg: DictConfig):
    super().__init__()
    self.self_attn = TimesFMAttention( cfg )
    self.mlp = TransformerMLP( cfg )
    self.input_layernorm = RMSNorm( cfg )

  def forward(self, hidden_states: torch.Tensor ) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn( hidden_states=hidden_states )
    hidden_states = residual + hidden_states
    hidden_states = self.mlp( hidden_states )
    return hidden_states

class StackedDecoder(nn.Module):

  def __init__(self, cfg: DictConfig):
    super().__init__()
    self.layers = nn.ModuleList()
    for _ in range(cfg.num_layers):
      self.layers.append( TimesFMDecoderLayer(cfg) )

  def forward( self, hidden_states: torch.Tensor ) -> torch.Tensor:
    for i in range(len(self.layers)):
      layer = self.layers[i]
      hidden_states = layer( hidden_states=hidden_states )
    return hidden_states
