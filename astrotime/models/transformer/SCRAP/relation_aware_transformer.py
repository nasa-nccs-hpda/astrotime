import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.svm import SVR
from typing import Any, Dict, List, Optional, Tuple
from .time_embedding import RelativePosition
from .attention import RelationAwareAttentionHead

# Set a random seed for reproducibility
torch.manual_seed(42)

class RelationAwareTransformer(nn.Module):


	def __init__(self, cfg: DictConfig ):
		super().__init__()
		self.relative_position_k = RelativePosition( cfg )
		self.relative_position_v = RelativePosition( cfg )
		self.k_bias_matrix = self.relative_position_k( cfg.seq_len, cfg.seq_len )
		self.v_bias_matrix = self.relative_position_v( cfg.seq_len, cfg.seq_len )
		self.attention_head = RelationAwareAttentionHead( self.k_bias_matrix, self.v_bias_matrix, cfg )
		self.attention_heads: nn.ModuleList = nn.ModuleList([RelationAwareAttentionHead( self.k_bias_matrix, self.v_bias_matrix, cfg ) for _ in range(cfg.num_heads)])
		self.fc: nn.Linear = nn.Linear(cfg.hidden_size, cfg.hidden_size)
		self.svr = SVR()


	def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
		"""
		Applies multi-head attention mechanism to the input query, key, and value tensors.

		Args:
			query (torch.Tensor): Query tensor.
			key (torch.Tensor): Key tensor.
			value (torch.Tensor): Value tensor.
			mask (torch.Tensor): Optional mask tensor.

		Returns:
			torch.Tensor: Updated hidden state after applying multi-head attention mechanism.
		"""
		attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
		hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
		hidden_state: torch.Tensor = self.fc(hidden_state)
		return hidden_state

# # Generate dummy input tensors
# x_input = torch.rand((batch_size, seq_len, hidden_size))
#
# # Test RelativePosition
# relative_position_embeddings = relative_position_k(seq_len, seq_len)
# print("Relative Position Embeddings Shape:", relative_position_embeddings.shape)
#
# # Test RelationAwareAttentionHead
# output_attention_head = attention_head(x_input, x_input, x_input)
# print("RelationAwareAttentionHead Output Shape:", output_attention_head.shape)
#
# # Test RelationAwareMultiHeadAttention
# output_multihead_attention = multihead_attention(x_input, x_input, x_input)
# print("RelationAwareMultiHeadAttention Output Shape:", output_multihead_attention.shape)
