import torch
import torch.nn as nn
from omegaconf import DictConfig
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

class RelationAwareAttentionHead(nn.Module):
    """
    Relation-aware attention head implementation.

    Args:
        k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
        v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.

    Attributes:
        query_weights (nn.Linear): Linear layer for query projection.
        key_weights (nn.Linear): Linear layer for key projection.
        value_weights (nn.Linear): Linear layer for value projection.
    """

    def __init__(self, k_bias_matrix, v_bias_matrix, cfg: DictConfig ):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.query_weights: nn.Linear  = nn.Linear(cfg.hidden_size, cfg.head_dim)
        self.key_weights:   nn.Linear  = nn.Linear(cfg.hidden_size, cfg.head_dim)
        self.value_weights: nn.Linear  = nn.Linear(cfg.hidden_size, cfg.head_dim)
        self.k_bias_matrix = k_bias_matrix
        self.v_bias_matrix = v_bias_matrix

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated value embeddings after applying attention mechanism.
        """
        query: torch.Tensor = self.query_weights(query) # (b_s, n_t, head_dim)
        key: torch.Tensor = self.key_weights(key) # (b_s, n_t, head_dim)
        value: torch.Tensor = self.value_weights(value) # (b_s, n_t, head_dim)

        # Self-Attention scores
        attn_1: torch.Tensor = torch.matmul(query, key.transpose(1, 2)) # Q*K^T:(b_s, n_t, n_t)

        # Relative Position Attention scores
        attn_2: torch.Tensor = torch.matmul(query.permute(1, 0, 2), self.k_bias_matrix.transpose(1, 2)).transpose(0, 1) # Q*K_shifting^T:(b_s, n_t, n_t)

        # Relation-aware Self-Attention scores
        att_scores: torch.Tensor = (attn_1 + attn_2)/self.head_dim ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores: torch.Tensor = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)

        # Weighted sum of values
        values_1: torch.Tensor = torch.matmul(att_weights, value) # (b_s, n_t, head_dim)

        # Relative Position Representation for values
        values_2: torch.Tensor = torch.matmul(att_weights.permute(1, 0, 2), self.v_bias_matrix).transpose(0, 1) # (b_s, n_t, head_dim)

        # Relation-aware values
        n_value  = values_1 + values_2
        return n_value

#
# class RelationAwareMultiHeadAttention(nn.Module):
#     """
#     Attributes:
#         hidden_size (int): Hidden size for the model (embedding dimension).
#         num_heads (int): Number of attention heads.
#         head_dim (int): Dimensionality of each attention head.
#         relative_position_k (RelativePosition): Instance of RelativePosition for query-key relative positions.
#         relative_position_v (RelativePosition): Instance of RelativePosition for query-value relative positions.
#         k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
#         v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.
#         attention_heads (nn.ModuleList): List of RelationAwareAttentionHead layers.
#         fc (nn.Linear): Fully connected layer for final projection.
#     """
#
#     def __init__(self, time_embeddings: Dict[str,nn.Module], cfg: DictConfig ):
#         super().__init__()
#         self.hidden_size: int = cfg.hidden_size
#         self.num_heads: int = cfg.num_heads
#         self.head_dim: int = cfg.hidden_size // cfg.num_heads
#         self.relative_position_k: torch.Tensor = RelativePosition(self.head_dim, cfg.k)
#         self.relative_position_v: torch.Tensor = RelativePosition(self.head_dim, cfg.k)
#         self.k_bias_matrix: torch.Tensor = self.relative_position_k(cfg.seq_len, cfg.seq_len)
#         self.v_bias_matrix: torch.Tensor = self.relative_position_v(cfg.seq_len, cfg.seq_len)
#         self.attention_heads: nn.ModuleList = nn.ModuleList([RelationAwareAttentionHead(self.hidden_size, self.head_dim,
#                                                                            self.k_bias_matrix, self.v_bias_matrix) for _ in range(self.num_heads)])
#         self.fc: nn.Linear = nn.Linear(cfg.hidden_size, cfg.hidden_size)
#
#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
#         """
#         Applies multi-head attention mechanism to the input query, key, and value tensors.
#
#         Args:
#             query (torch.Tensor): Query tensor.
#             key (torch.Tensor): Key tensor.
#             value (torch.Tensor): Value tensor.
#             mask (torch.Tensor): Optional mask tensor.
#
#         Returns:
#             torch.Tensor: Updated hidden state after applying multi-head attention mechanism.
#         """
#         attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
#         hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
#         hidden_state: torch.Tensor = self.fc(hidden_state)
#         return hidden_state
