import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Mapping

def get_model( **kwargs ) -> nn.Module:
	model_layers: List[nn.Module] = [
		nn.Conv1d(68, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(72, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(76, kernel_size=3, activation='elu', padding='same'),
		nn.BatchNorm1d(),
		nn.MaxPool1d(2),
		nn.Conv1d(80, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(84, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(88, kernel_size=3, activation='elu', padding='same'),
		nn.BatchNorm1d(),
		nn.MaxPool1d(2),
		nn.Conv1d(92, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(96, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(100, kernel_size=3, activation='elu', padding='same'),
		nn.BatchNorm1d(),
		nn.MaxPool1d(2),
		nn.Conv1d(104, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(108, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(112, kernel_size=3, activation='elu', padding='same'),
		nn.BatchNorm1d(),
		nn.MaxPool1d(2),
		nn.Conv1d(116, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(120, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(124, kernel_size=3, activation='elu', padding='same'),
		nn.BatchNorm1d(),
		nn.MaxPool1d(2),
		nn.Conv1d(128, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(132, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(136, kernel_size=3, activation='elu', padding='same'),
		nn.BatchNorm1d(),
		nn.MaxPool1d(2),
		nn.Conv1d(140, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(144, kernel_size=3, activation='elu', padding='same'),
		nn.Conv1d(148, kernel_size=3, activation='elu', padding='same'),
		nn.BatchNorm1d(),
		nn.MaxPool1d(2),
		nn.Flatten(),
		nn.Linear(64, activation='elu'),
		nn.Linear(1)
	]
	model = nn.Sequential(model_layers)
	return model