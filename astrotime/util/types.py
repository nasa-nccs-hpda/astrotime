from typing import TYPE_CHECKING, Any, Callable, TypedDict, TypeVar, Union

# External libraries
# Matplotlib
from matplotlib.axes import Axes as MatplotlibAxes
from matplotlib.figure import Figure as MatplotlibFigure

# Numpy
from numpy.typing import ArrayLike as NumpyArrayLike

# Pandas
from pandas import Timestamp

# External Types
Axes = TypeVar("Axes", bound=MatplotlibAxes)  # Matplotlib Axes
Figure = TypeVar("Figure", bound=MatplotlibFigure)  # Matplotlib Figure
ArrayLike = TypeVar("ArrayLike", bound=NumpyArrayLike)  # Array Like (NumPy based)
