import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, device

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.
    """

    def __init__( self, cfg: DictConfig, device: device ):
        factory_kwargs = {"device": device, "dtype": None}
        super().__init__()
        self.cfg = cfg
        self.nheads: int = cfg.nheads
        self.dropout: float = cfg.dropout
        self._qkv_same_embed_dim: bool = cfg.E_q == cfg.E_k and cfg.E_q == cfg.E_v

        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(cfg.E_q, cfg.E_hidden * 3, bias=cfg.bias, **factory_kwargs)
        else:
            self.q_proj: nn.Module = nn.Linear(cfg.E_q, cfg.E_hidden, bias=cfg.bias, **factory_kwargs)
            self.k_proj: nn.Module = nn.Linear(cfg.E_k, cfg.E_hidden, bias=cfg.bias, **factory_kwargs)
            self.v_proj: nn.Module = nn.Linear(cfg.E_v, cfg.E_hidden, bias=cfg.bias, **factory_kwargs)
        self.out_proj: nn.Module = nn.Linear(cfg.E_hidden, cfg.E_out, bias=cfg.bias, **factory_kwargs)

        assert cfg.E_hidden % cfg.nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head: int = cfg.E_hidden // cfg.nheads
        self.bias: bool = cfg.bias

    def forward( self, query: Tensor, key: Tensor, value: Tensor ) -> Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (Tensor): value of shape (``N``, ``L_kv``, ``E_v``)

        Returns:
            attn_output (Tensor): output of shape (N, L_t, E_out)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_hidden) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query: Tensor = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_hidden) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key: Tensor = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_hidden) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value: Tensor = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention( query, key, value, dropout_p=self.dropout )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_hidden)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_hidden) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output