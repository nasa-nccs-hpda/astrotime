import numpy as np

def logspace(start: float, stop: float, N: int) -> np.ndarray:
	return np.pow(10.0, np.linspace(np.log10(start), np.log10(stop), N))