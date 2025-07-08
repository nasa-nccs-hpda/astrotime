import torch, logging
import torch.nn as nn
from astrotime.util.math import shp
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from astrotime.util.tensor_ops import check_nan
from torch import Tensor, device

class MultiHeadAttention(nn.Module):

    def __init__( self, cfg: DictConfig, device: device, input_size: int, output_size: int, **kwargs ):
        factory_kwargs = {"device": device, "dtype": None}
        super().__init__()
        self.log = logging.getLogger()
        self.cfg = cfg
        self.nheads: int = cfg.nheads
        self.dropout: float = cfg.dropout
        self.E_head: int = cfg.E_head
        E_total = self.nheads * self.E_head
        self.packed_proj = nn.Linear( input_size, E_total * 3, bias=cfg.bias, **factory_kwargs )
        self.out_proj: nn.Module = nn.Linear(E_total, output_size, bias=cfg.bias, **factory_kwargs)
        self.bias: bool = cfg.bias
        self.verbose = kwargs.get("verbose", False)
        if self.verbose: print(f" MultiHeadAttention ----> input_size={input_size} output_size={output_size} nheads={self.nheads} E_head={self.E_head} proj_size={E_total} packed_proj_size={E_total*3} ")

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

        if self.verbose: check_nan(f"packed_proj.input", embedding)
        result = self.packed_proj(embedding)

        if self.verbose: check_nan(f"packed_proj.result", result)
        query, key, value = torch.chunk(result, 3, dim=-1)

        if self.verbose: print(f" embedding{shp(embedding)} ----> (N, L_t, E_hidden): query{shp(query)} key{shp(key)} value{shp(value)}") #
        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_hidden) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query: Tensor = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_hidden) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key: Tensor = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_hidden) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value: Tensor = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        if self.verbose:
            check_nan( f"s2.query", query )
            check_nan( f"s2.key", key )
            check_nan( f"s2.value", value)
            print(f" ---->  (N, L_s, E_hidden) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head): query{shp(query)} key{shp(key)} value{shp(value)}")

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention( query, key, value, dropout_p=self.dropout )

        if self.verbose: check_nan( f"attn_output", attn_output)

        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_hidden)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        if self.verbose: print(f" ----> (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_hidden): attn_output{shp(attn_output)}")

        # Step 4. Apply output projection
        # (N, L_t, E_hidden) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        if self.verbose:  print(f" ----> (N, L_t, E_hidden) -> (N, L_t, E_out): attn_output{shp(attn_output)}")

        return attn_output