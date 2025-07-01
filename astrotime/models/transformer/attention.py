import torch
import torch.nn as nn
from astrotime.util.math import shp
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, device

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.
    """

    def __init__( self, cfg: DictConfig, device: device, embedding_size: int ):
        factory_kwargs = {"device": device, "dtype": None}
        super().__init__()
        self.cfg = cfg
        self.nheads: int = cfg.nheads
        self.dropout: float = cfg.dropout
        self.E_head: int = cfg.E_head
        E_total = self.nheads * self.E_head
        self.packed_proj = nn.Linear( embedding_size, E_total * 3, bias=cfg.bias, **factory_kwargs )
        self.out_proj: nn.Module = nn.Linear(E_total, cfg.E_out, bias=cfg.bias, **factory_kwargs)
        self.bias: bool = cfg.bias

    def forward( self, embedding: Tensor ) -> Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Returns:
            attn_output (Tensor): output of shape (N, L_t, E_out)
        """
        # Step 1. Apply input projection

        self.log.debug(f" ----> s0: embedding{shp(embedding)}")

        result = self.packed_proj(embedding)
        query, key, value = torch.chunk(result, 3, dim=-1)

        self.log.debug(f" ----> s1: query{shp(query)} key{shp(key)} value{shp(value)}")
        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_hidden) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query: Tensor = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_hidden) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key: Tensor = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_hidden) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value: Tensor = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        self.log.debug(f" ----> s2: query{shp(query)} key{shp(key)} value{shp(value)}")

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention( query, key, value, dropout_p=self.dropout )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_hidden)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        self.log.debug(f" ----> s1: attn_output{shp(attn_output)}")

        # Step 4. Apply output projection
        # (N, L_t, E_hidden) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        self.log.debug(f" ----> s2: attn_output{shp(attn_output)}")

        return attn_output