import logging, numpy as np
from astrotime.util.logging import exception_handled
from matplotlib.widgets import Button, RadioButtons, TextBox, Slider
#from astrotime.plot.widgets import Slider
from typing import Dict, List, Tuple, Type, Callable

Number = float | int
Parameter = Number | str

class STParam:

	def __init__(self, name: str, value: Parameter ) -> None:
		self.name: str = name
		self.value: Parameter = value
		self.aux_sizes = None
		self.log = logging.getLogger()

	def widget( self, ax, callbacks: List[Callable], aux_axes=None ):
		raise NotImplementedError( "The abstract method 'widget' of class 'STParam' is not implemented.")

	def set_value(self, value: Parameter):
		self.value = value

	def value_selected(self):
		return self.value

	def process_key_press(self, key: str ):
		pass

	def __repr__(self) -> str:
		val = f"{self.value:.3f}" if type(self.value) is float else f"{self.value}"
		return f"{self.name}: {val}"

	def __str__(self) -> str:
		return self.__repr__()

class Parameterized(object):

	def __init__(self):
		self._sparms: Dict[str, STParam] = {}
		self._current_value: Dict[str, Parameter] = {}
		self.param_domain: np.ndarray = np.linspace(0, 1.0, 100)
		self.children: List[Parameterized] = []
		self.log = logging.getLogger()

	def share_param(self, param: STParam):
		if param.name in self._sparms:
	#		log.info( f" {type(self).__name__}: Sharing parameter '{param.name}' ({hex(id(param))}) with {len(self.children)} children")
			self.add_param( param )
			for child in self.children:
				child.share_param( param )

	@property
	def sparms(self) -> Dict[str, STParam]:
		return self._sparms

	def sval(self,name) -> Number:
		self.pname_check(name)
		param = self._sparms[name]
		return param.value_selected()

	def set_param(self, name, value:Parameter ):
		self.pname_check(name)
		param = self._sparms[name]
		return param.set_value(value)

	def __getattr__(self, name):
		if name in self._sparms:
			param = self._sparms[name]
			return param.value_selected()
		return super().__getattribute__(name)

	def pname_check(self, pname: str):
		assert pname in self._sparms, f"Unrecognized parameter in SignalTransform: '{pname}', known parameters: {list(self._sparms.keys())}"

	def add_param(self, p: STParam):
		self._sparms[p.name] = p
		self._current_value[p.name] = p.value

	def needs_refresh(self, plist: List[str] = None) -> bool:
		if plist is None: plist = self._sparms.keys()
		for pn in plist:
			if self._current_value[pn] != self.sparms[pn].value_selected():
				return True
		return False

	def update_parms(self, plist: List[str] = None):
		if plist is None: plist = self._sparms.keys()
		for pn in plist:
			self._current_value[pn] = self.sparms[pn].value_selected()

	def inherit_params(self, parameterized: "Parameterized" ):
		for p in parameterized.sparms.values(): self.add_param(p)
		self.children.append( parameterized )

class LogScaleSlider(Slider):

	def __init__(self, ax, label, valmin, valmax, *args, **kwargs):
		valinit = kwargs.pop('valinit', valmin)
		lvalmin, lvalmax = np.log10(valmin), np.log10(valmax)
		super(LogScaleSlider, self).__init__( ax, label, lvalmin, lvalmax, *args, valinit=np.log10(valinit), **kwargs)
		self.val = valinit

	def set_val(self, val):
		if self.orientation == 'vertical':
			self.poly.set_height(val - self.poly.get_y())
			self._handle.set_ydata([val])
		else:
			self.poly.set_width(val - self.poly.get_x())
			self._handle.set_xdata([val])
		lval = 10**val
		self.valtext.set_val(self._format(lval))
		if self.drawon:
			self.ax.figure.canvas.draw_idle()
		self.val = lval
		if self.eventson:
			self._observers.process('changed', lval)

class FloatValuesSlider(Slider):

	def __init__(self, ax, label, values: List[float], *args, **kwargs):
		valinit = kwargs.pop('valinit', 0)
		self.values: List[float] = values
		self.val = valinit
		super(FloatValuesSlider, self).__init__( ax, label, 0, len(values)-1, *args, valinit=valinit, **kwargs)

	def set_val(self, ival):
		if self.orientation == 'vertical':
			self.poly.set_height(ival - self.poly.get_y())
			self._handle.set_ydata([ival])
		else:
			self.poly.set_width(ival - self.poly.get_x())
			self._handle.set_xdata([ival])
		lval = self.values[ival]
		self.valtext.set_val(self._format(lval))
		if self.drawon:
			self.ax.figure.canvas.draw_idle()
		self.val = lval
		if self.eventson:
			self._observers.process('changed', lval)

	def value_selected(self):
		return self.val

class STIntParam(STParam):

	def __init__(self, name: str, vrange: Tuple[int,int] = (0,100), **kwargs ) -> None:
		super(STIntParam, self).__init__( name, kwargs.get('value', vrange[0]))
		self.vrange: Tuple[int, int] = vrange
		self.step: int = kwargs.get( 'step', 1 )
		self._widget: Slider = None
		self.log.info( f" STIntParam: vrange = {self.vrange}" )
		self.key_press_mode = kwargs.get('key_press_mode', 1)

	def process_key_press(self, key: str ):
		if self.key_press_mode == 1:
			if   key == "right": self.set_value( self.value_selected() + 1 )
			elif key == "left":  self.set_value( self.value_selected() - 1 )
		elif self.key_press_mode == 2:
			if key == "up":      self.set_value( self.value_selected() + 1 )
			elif key == "down":  self.set_value( self.value_selected() - 1 )

	@exception_handled
	def set_value(self , val):
		self._widget.set_val( int(val) )
		self._widget.canvas.draw_idle()

	def widget(self, ax, callbacks: List[Callable], aux_axes=None ):
		if self._widget is None:
			print( [ self.name, self.vrange[0], self.vrange[1], self.value, self.step ] )
			self._widget = Slider( ax, self.name, self.vrange[0], self.vrange[1], valinit=self.value, valstep=self.step )
			for callback in callbacks:
				self._widget.on_changed(callback)
		return self._widget

	def value_selected(self):
		return self.value if (self._widget is None) else self._widget.val


class STFloatValuesParam(STParam):

	def __init__(self, name: str, values: List[float], rval: Type[float]|Type[int], **kwargs) -> None:
		self.values: List[float] = values
		self.vrange: Tuple[int, int] = (0,len(values)-1)
		super(STFloatValuesParam, self).__init__( name, kwargs.get('value', self.vrange[0]) )
		self.rval: Type[float]|Type[int] = rval
		self.step: int = kwargs.get( 'step', 1 )
		self._widget: Slider = None

	def widget(self, ax, callbacks: List[Callable], aux_axes=None ):
		if self._widget is None:
			self._widget = FloatValuesSlider( ax, self.name, self.values, valinit=self.value, valstep=self.step )
			for callback in callbacks:
				self._widget.on_changed(callback)
		return self._widget

	def value_selected(self):
		if self._widget is None:
			return self.value
		else:
			return self._widget.val if (self.rval is float) else self.values.index(self._widget.val)

class STFloatParam(STParam):

	def __init__(self, name: str, vrange: Tuple[float, float] = (0.0,1.0), **kwargs ) -> None:
		self._log = kwargs.get('log', False)
		ival = kwargs.get('value', vrange[0])
		super(STFloatParam, self).__init__( name, ival )
		self.vrange: Tuple[float, float] = vrange
		self.step: float = kwargs.get('step', (vrange[1]-vrange[0])/100 )
		self._widget: Slider = None

	def widget(self, ax, callbacks: List[Callable], aux_axes=None ):
		if self._widget is None:
			slider_cls = LogScaleSlider if self._log else Slider
			self._widget = slider_cls( ax, self.name, self.vrange[0], self.vrange[1], valinit=self.value, valstep=self.step )
			for callback in callbacks:
				self._widget.on_changed(callback)
		return self._widget

	def value_selected(self):
		return self.value if (self._widget is None) else self._widget.val

class STStrParam(STParam):

	def __init__(self, name: str, values: List[str], **kwargs  ) -> None:
		super(STStrParam, self).__init__( name, kwargs.get('value', values[0]) )
		self.values: List[str] = values
		assert self.value in self.values, f"Unknown value for STParam: {self.value}"
		self._widget: RadioButtons = None

	def widget(self, ax, callbacks: List[Callable], aux_axes=None  ):
		self._widget = RadioButtons(ax, self.values, activecolor="yellow" )
		for callback in callbacks:
			self._widget.on_clicked(callback)
		return self._widget

	def value_selected(self):
		return self.value if (self._widget is None) else self._widget.value_selected