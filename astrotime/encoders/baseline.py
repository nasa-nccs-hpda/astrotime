import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Optional
from astrotime.encoders.base import Encoder

class ValueEncoder(Encoder):

	def __init__(self, series_len: int):
		super().__init__()
		self.series_len = series_len

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> np.ndarray:
		val_Xs = []
		for s in dset['y']:
			scaler = MinMaxScaler()
			val_Xs.append(scaler.fit_transform(s[:self.series_len].reshape(-1, 1))[:, 0])
		result = np.array(val_Xs)
		return result