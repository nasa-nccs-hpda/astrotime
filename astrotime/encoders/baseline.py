import random, numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Optional
from astrotime.encoders.base import Encoder

class ValueEncoder(Encoder):

	def __init__(self, series_len: int):
		super().__init__()
		self.series_len = series_len
		self.slmax = 6000

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> np.ndarray:
		val_Xs = []
		for s in dset['y']:
			scaler = MinMaxScaler()
			t0 = random.randrange(0, self.slmax - self.series_len )
			val_Xs.append( scaler.fit_transform( s[t0:t0+self.series_len].reshape(-1, 1) ) )
		result = np.array(val_Xs)
		return result