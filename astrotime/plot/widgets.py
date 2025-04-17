from matplotlib.lines import Line2D
from contextlib import ExitStack
import matplotlib as mpl
from matplotlib import (_api, _docstring, backend_tools, cbook, collections, colors,  text as mtext, ticker, transforms)
import logging
from matplotlib.patches import Rectangle, Ellipse, Polygon
from matplotlib.transforms import TransformedPatchPath, Affine2D
from matplotlib.widgets import SliderBase, AxesWidget
from matplotlib.axes import Axes

class Slider(SliderBase):
    """
    A slider representing a floating point range.

    Create a slider from *valmin* to *valmax* in Axes *ax*. For the slider to
    remain responsive you must maintain a reference to it. Call
    :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : float
        Slider value.
    """

    def __init__(self, ax: Axes, label: str, valmin, valmax, *, valinit=0.5, valfmt=None,
                 closedmin=True, closedmax=True, slidermin=None,
                 slidermax=None, dragging=True, valstep=None,
                 orientation='horizontal', initcolor='r',
                 track_color='lightgrey', handle_style=None, **kwargs):
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : float, default: 0.5
            The slider initial position.

        valfmt : str, default: None
            %-format string used to format the slider value.  If None, a
            `.ScalarFormatter` is used instead.

        closedmin : bool, default: True
            Whether the slider interval is closed on the bottom.

        closedmax : bool, default: True
            Whether the slider interval is closed on the top.

        slidermin : Slider, default: None
            Do not allow the current slider to have a value less than
            the value of the Slider *slidermin*.

        slidermax : Slider, default: None
            Do not allow the current slider to have a value greater than
            the value of the Slider *slidermax*.

        dragging : bool, default: True
            If True the slider can be dragged by the mouse.

        valstep : float or array-like, default: None
            If a float, the slider will snap to multiples of *valstep*.
            If an array the slider will snap to the values in the array.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The orientation of the slider.

        initcolor : :mpltype:`color`, default: 'r'
            The color of the line at the *valinit* position. Set to ``'none'``
            for no line.

        track_color : :mpltype:`color`, default: 'lightgrey'
            The color of the background track. The track is accessible for
            further styling via the *track* attribute.

        handle_style : dict
            Properties of the slider handle. Default values are

            ========= ===== ======= ========================================
            Key       Value Default Description
            ========= ===== ======= ========================================
            facecolor color 'white' The facecolor of the slider handle.
            edgecolor color '.75'   The edgecolor of the slider handle.
            size      int   10      The size of the slider handle in points.
            ========= ===== ======= ========================================

            Other values will be transformed as marker{foo} and passed to the
            `~.Line2D` constructor. e.g. ``handle_style = {'style'='x'}`` will
            result in ``markerstyle = 'x'``.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        `~matplotlib.patches.Rectangle` that draws the slider knob.  See the
        `.Rectangle` documentation for valid property names (``facecolor``,
        ``edgecolor``, ``alpha``, etc.).
        """
        super().__init__(ax, orientation, closedmin, closedmax, valmin, valmax, valfmt, dragging, valstep)
        self.log = logging.getLogger()
        if slidermin is not None and not hasattr(slidermin, 'val'):
            raise ValueError( f"Argument slidermin ({type(slidermin)}) has no 'val'")
        if slidermax is not None and not hasattr(slidermax, 'val'):
            raise ValueError( f"Argument slidermax ({type(slidermax)}) has no 'val'")
        self.slidermin = slidermin
        self.slidermax = slidermax
        valinit = self._value_in_bounds(valinit)
        if valinit is None:
            valinit = valmin
        self.val = valinit
        self.valinit = valinit

        defaults = {'facecolor': 'white', 'edgecolor': '.75', 'size': 10}
        handle_style = {} if handle_style is None else handle_style
        marker_props = { f'marker{k}': v for k, v in {**defaults, **handle_style}.items()  }
        self.track = Rectangle(
            (0, .25), 1, .5,
            transform=ax.transAxes,
            facecolor=track_color
        )
        ax.add_patch(self.track)
        self.poly = ax.axvspan(valmin, valinit, .25, .75, **kwargs)
        self.vline = ax.axvline(valinit, 0, 1, color=initcolor, lw=1,  clip_path=TransformedPatchPath(self.track))
        handleXY = [[valinit], [0.5]]
        self._handle, = ax.plot(  *handleXY,  "o", **marker_props,  clip_on=False )
        self.valtext = TextBox( ax,1.02, 0.5, str(valinit) )
        self.set_val(valinit)

    def _value_in_bounds(self, val):
        """Makes sure *val* is with given bounds."""
        val = self._stepped_value(val)

        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val
        return val

    def _update(self, event):
        """Update the slider position."""
        if event.name != 'motion_notify_event':
            self.log.info( f" *** Slider Event: {event}")

        if self.ignore(event) or event.button != 1:
            return

        if event.name == 'button_press_event' and self.ax.contains(event)[0]:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        if event.name == 'button_release_event' or event.name == 'button_press_event' and not self.ax.contains(event)[0]:
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return

        xdata, ydata = self._get_data_coords(event)
        val = self._value_in_bounds(
            xdata if self.orientation == 'horizontal' else ydata)
        if val not in [None, self.val]:
            self.set_val(val)

    def _format(self, val):
        """Pretty-print *val*."""
        if self.valfmt is not None:
            return self.valfmt % val
        else:
            _, s, _ = self._fmt.format_ticks([self.valmin, val, self.valmax])
            # fmt.get_offset is actually the multiplicative factor, if any.
            return s + self._fmt.get_offset()

    def set_text(self, val):
        return self.set_val(val)

    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : float
        """

        self.poly.set_width(val - self.poly.get_x())
        self._handle.set_xdata([val])
        self.valtext.set_val(str(val))
        if self.drawon:
            self.ax.get_figure(root=True).canvas.draw_idle()
        self.val = val
        if self.eventson:
            self._observers.process('changed', val)

    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed.
            The function must accept a single float as its arguments.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect('changed', lambda val: func(val))


class TextBox(AxesWidget):
    """
    Call `.on_text_change` to be updated whenever the text changes.
    Call `.on_submit` to be updated whenever the user hits enter or leaves the text entry field.
    """

    def __init__(self, ax, x, y, initial='' ):

        super().__init__(ax)
        self.text_disp = self.ax.text( x, y, initial, transform=self.ax.transAxes, verticalalignment='center', horizontalalignment='left', parse_math=False)
        self._observers = cbook.CallbackRegistry(signals=["change", "submit"])
        #ax.set( xlim=(0, 1), ylim=(0, 1), navigate=False, facecolor="darkturquoise", xticks=[], yticks=[])

        self.cursor_index = 0
        self.cursor = ax.vlines(0, 0, 0, visible=False, color="k", lw=1,  transform=mpl.transforms.IdentityTransform())

        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)
        self.connect_event('key_press_event', self._keypress)
        self.connect_event('resize_event', self._resize)

        self.color = 'lightblue'
        self.hovercolor = 'skyblue'

        self.capturekeystrokes = False

    @property
    def text(self):
        return self.text_disp.get_text()

    def _rendercursor(self):
        # this is a hack to figure out where the cursor should go.
        # we draw the text up to where the cursor should go, measure
        # and save its dimensions, draw the real text, then put the cursor
        # at the saved dimensions

        # This causes a single extra draw if the figure has never been rendered
        # yet, which should be fine as we're going to repeatedly re-render the
        # figure later anyways.
        fig = self.ax.get_figure(root=True)
        if fig._get_renderer() is None:
            fig.canvas.draw()

        text = self.text_disp.get_text()  # Save value before overwriting it.
        widthtext = text[:self.cursor_index]

        bb_text = self.text_disp.get_window_extent()
        self.text_disp.set_text(widthtext or ",")
        bb_widthtext = self.text_disp.get_window_extent()

        if bb_text.y0 == bb_text.y1:  # Restoring the height if no text.
            bb_text.y0 -= bb_widthtext.height / 2
            bb_text.y1 += bb_widthtext.height / 2
        elif not widthtext:  # Keep width to 0.
            bb_text.x1 = bb_text.x0
        else:  # Move the cursor using width of bb_widthtext.
            bb_text.x1 = bb_text.x0 + bb_widthtext.width

        self.cursor.set(
            segments=[[(bb_text.x1, bb_text.y0), (bb_text.x1, bb_text.y1)]],
            visible=True)
        self.text_disp.set_text(text)

        fig.canvas.draw()

    def _release(self, event):
        if self.ignore(event):
            return
        if event.canvas.mouse_grabber != self.ax:
            return
        event.canvas.release_mouse(self.ax)

    def _keypress(self, event):
        if self.ignore(event):
            return
        if self.capturekeystrokes:
            key = event.key
            text = self.text
            if len(key) == 1:
                text = (text[:self.cursor_index] + key +
                        text[self.cursor_index:])
                self.cursor_index += 1
            elif key == "right":
                if self.cursor_index != len(text):
                    self.cursor_index += 1
            elif key == "left":
                if self.cursor_index != 0:
                    self.cursor_index -= 1
            elif key == "home":
                self.cursor_index = 0
            elif key == "end":
                self.cursor_index = len(text)
            elif key == "backspace":
                if self.cursor_index != 0:
                    text = (text[:self.cursor_index - 1] +
                            text[self.cursor_index:])
                    self.cursor_index -= 1
            elif key == "delete":
                if self.cursor_index != len(self.text):
                    text = (text[:self.cursor_index] +
                            text[self.cursor_index + 1:])
            self.text_disp.set_text(text)
            self._rendercursor()
            if self.eventson:
                self._observers.process('change', self.text)
                if key in ["enter", "return"]:
                    self._observers.process('submit', self.text)

    def set_val(self, val):
        newval = str(val)
        if self.text == newval:
            return
        self.text_disp.set_text(newval)
        self._rendercursor()
        if self.eventson:
            self._observers.process('change', self.text)
            self._observers.process('submit', self.text)

    def begin_typing(self):
        self.capturekeystrokes = True
        # Disable keypress shortcuts, which may otherwise cause the figure to
        # be saved, closed, etc., until the user stops typing.  The way to
        # achieve this depends on whether toolmanager is in use.
        stack = ExitStack()  # Register cleanup actions when user stops typing.
        self._on_stop_typing = stack.close
        toolmanager = getattr(
            self.ax.get_figure(root=True).canvas.manager, "toolmanager", None)
        if toolmanager is not None:
            # If using toolmanager, lock keypresses, and plan to release the
            # lock when typing stops.
            toolmanager.keypresslock(self)
            stack.callback(toolmanager.keypresslock.release, self)
        else:
            # If not using toolmanager, disable all keypress-related rcParams.
            # Avoid spurious warnings if keymaps are getting deprecated.
            with _api.suppress_matplotlib_deprecation_warning():
                stack.enter_context(mpl.rc_context(  {k: [] for k in mpl.rcParams if k.startswith("keymap.")}))

    def stop_typing(self):
        if self.capturekeystrokes:
            self._on_stop_typing()
            self._on_stop_typing = None
            notifysubmit = True
        else:
            notifysubmit = False
        self.capturekeystrokes = False
        self.cursor.set_visible(False)
        self.ax.get_figure(root=True).canvas.draw()
        if notifysubmit and self.eventson:
            # Because process() might throw an error in the user's code, only
            # call it once we've already done our cleanup.
            self._observers.process('submit', self.text)

    def _click(self, event):
        if self.ignore(event):
            return
        if not self.ax.contains(event)[0]:
            self.stop_typing()
            return
        if not self.eventson:
            return
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)
        if not self.capturekeystrokes:
            self.begin_typing()
        self.cursor_index = self.text_disp._char_index_at(event.x)
        self._rendercursor()

    def _resize(self, event):
        self.stop_typing()

    def _motion(self, event):
        if self.ignore(event):
            return
        c = self.hovercolor if self.ax.contains(event)[0] else self.color
        if not colors.same_color(c, self.ax.get_facecolor()):
            self.ax.set_facecolor(c)
            if self.drawon:
                self.ax.get_figure(root=True).canvas.draw()

    def on_text_change(self, func):
        """
        When the text changes, call this *func* with event.

        A connection id is returned which can be used to disconnect.
        """
        return self._observers.connect('change', lambda text: func(text))

    def on_submit(self, func):
        """
        When the user hits enter or leaves the submission box, call this
        *func* with event.

        A connection id is returned which can be used to disconnect.
        """
        return self._observers.connect('submit', lambda text: func(text))

    def disconnect(self, cid):
        """Remove the observer with connection id *cid*."""
        self._observers.disconnect(cid)
