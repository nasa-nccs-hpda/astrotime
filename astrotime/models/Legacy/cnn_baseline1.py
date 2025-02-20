from fusion.config.context import ConfigContext, cfg
from fusion.models.common.common import BaseModel
from fusion.base.util.logging import lgm, exception_handled, log_timing, shp
import time, numpy as np
from fusion.models.common.common import init_parms, cnn_series_length
import torch
from typing import Any, Dict, List, Tuple
import torch.nn as nn

class CNNBlock(BaseModel):
	def __init__(self, in_channels:int, out_channels:int, device: torch.device, mparms: Dict[str,Any], **custom_parms : Dict[str,Any] ):
		super(CNNBlock, self).__init__(device, mparms, **custom_parms)
		self.conv1 = nn.Conv1d( in_channels,    out_channels-8, kernel_size=self.kernel_size, stride=self.stride, padding='same').to(self.device)
		self.conv2 = nn.Conv1d( out_channels-8, out_channels-4, kernel_size=self.kernel_size, stride=self.stride, padding='same').to(self.device)
		self.conv3 = nn.Conv1d( out_channels-4, out_channels,   kernel_size=self.kernel_size, stride=self.stride, padding='same').to(self.device)
		self.batchnorm1 = nn.BatchNorm1d(out_channels).to(self.device)
		self.maxpool1 = nn.MaxPool1d(self.pool_size).to(self.device)
		self.act = nn.ELU().to(self.device)

	def forward(self, inputs):
		x = self.act( self.conv1(inputs) )
		x = self.act( self.conv2(x) )
		x = self.act( self.conv3(x) )
		x = self.batchnorm1(x)
		x = self.maxpool1(x)
		print( f"CNNBlock.forward: {shp(inputs)} -> {shp(x)}")
		return x

class DenseBlock(BaseModel):
	def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, device: torch.device, mparms: Dict[str,Any], **custom_parms : Dict[str,Any]  ):
		super(DenseBlock, self).__init__(device, mparms, **custom_parms)
		self.flatten = nn.Flatten().to(self.device)
		self.fc1 = nn.Linear( in_channels, hidden_channels).to(self.device)
		self.fc2 = nn.Linear( hidden_channels, out_channels).to(self.device)
		self.act = nn.ELU().to(self.device)

	def forward(self, inputs):
		x = self.flatten(inputs)
		x = self.act( self.fc1(x) )
		x =self.fc2(x)
		print(f"DenseBlock.forward: {shp(inputs)} -> {shp(x)}")
		return x

class SinusoidPeriodModel(BaseModel):
	def __init__(self, device: torch.device, mparms: Dict[str,Any], **custom_parms : Dict[str,Any]):
		super(SinusoidPeriodModel, self).__init__(device, mparms, **custom_parms)
		self.cnn_blocks: nn.ModuleList = nn.ModuleList()
		tlength = self.series_length
		for iL, cnn_size in enumerate(self.cnn_sizes):
			input_channels = self.nfeatures if iL == 0 else self.cnn_sizes[iL-1]
			tlength = tlength // self.pool_size
			self.cnn_blocks.append( CNNBlock(input_channels, cnn_size, device, mparms, **custom_parms ) )
		self.dense_block = DenseBlock( (self.cnn_sizes[-1])*tlength, self.dense_size, self.output_size, device, mparms, **custom_parms  )

	@torch.compile
	def forward(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
		x: torch.Tensor = batch
		for ilayer, layer in enumerate(self.cnn_blocks):
			x = layer(x)
		x = self.dense_block(x)
		return x


