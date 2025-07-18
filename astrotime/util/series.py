import numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from scipy.interpolate import CubicSpline, PchipInterpolator
import logging
from enum import Enum
log = logging.getLogger()

class TSet(Enum):
	Train = 'train'
	Validation = 'valid'
	Update = "update"

def series_interpolation(x: np.ndarray, y: np.ndarray, num_points: int, **kwargs ) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Computes cubic interpolation for an irregular timeseries and returns interpolated samples.

	Args:
		x (np.ndarray): The independent variable (irregularly spaced time data).
		y (np.ndarray): The dependent variable (values corresponding to time points).
		num_points (int): The number of points to interpolate over a regular grid

	Returns:
		Tuple[np.ndarray, np.ndarray]: A tuple containing the regular grid (x_interp)
										and the interpolated values (y_interp).

	Raises:
		ValueError: If x or y are empty, or sizes of x and y do not match.
	"""
	if x.size == 0 or y.size == 0:
		raise ValueError("Input arrays x and y cannot be empty.")
	if x.size != y.size:
		raise ValueError("Input arrays x and y must have the same length.")

	log.info( f"series_interpolation: {x.shape} {y.shape}")
	itype: str = kwargs.get( 'type', 'monotone' )
	x_interp = np.linspace(x.min(), x.max(), num_points)
	interp_function = CubicSpline(x, y) if (itype == 'spline') else PchipInterpolator(x, y)
	y_interp = interp_function(x_interp)
	return x_interp, y_interp

def get_peak( x: np.ndarray, y: np.ndarray, **kwargs ) -> Tuple[np.ndarray, np.ndarray, np.int32]:
	upsample: int = kwargs.get('upsample', 1)
	stype: str = kwargs.get('type', "spline")
	max_index: np.int32 = np.argmax(y)
	if upsample <= 1:
		return x, y, max_index
	else:
		peak_ext = kwargs.get('peak_ext', 3)
		prng = [ max( round(max_index-peak_ext), 0 ), min( round(max_index+peak_ext+1), x.size ) ]
		xp, yp = x[prng[0]:prng[1]], y[prng[0]:prng[1]]
		xpi, ypi = series_interpolation( xp, yp, round(upsample*xp.size), type=stype )
		max_index = np.argmax( ypi )
		return xpi, ypi, max_index