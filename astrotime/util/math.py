import numpy as np
from typing import List, Optional, Dict, Type, Any

def logspace(start: float, stop: float, N: int) -> np.ndarray:
	return np.pow(10.0, np.linspace(np.log10(start), np.log10(stop), N))

def shp( x ) -> List[int]:
	return list(x.shape)