import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Optional
from astrotime.encoders.base import Encoder

class ValueEncoder(Encoder):

	def __init__(self, series_len: int):
		super().__init__()
		self.series_len = series_len

	def encode_dset(self, batch_data: Dict):
		val_Xs = []
		for s in batch_data['y']:
			scaler = MinMaxScaler()
			val_Xs.append(scaler.fit_transform(s[:self.series_len].reshape(-1, 1))[:, 0])
		result = np.array(val_Xs)
		return result