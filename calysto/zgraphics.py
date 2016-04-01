from calysto.graphics import (Canvas, Color, Pixel, Point)

from ipywidgets import widgets
import time

from calysto.graphics import (Text      as _CText,
                              Rectangle as _CRectangle,
                              Ellipse   as _CEllipse,
                              Line      as _CLine,
                              Polyline  as _CPolyline,
                              Polygon   as _CPolygon,
                              Picture   as _CPicture,
                              Arc       as _CArc,
                              Circle    as _CCircle)

class GraphWin(Canvas):
    ID = 1
    def __init__(self, title="Graphics Window", width=320, height=240):
        super(GraphWin, self).__init__(size=(width, height))
        from calysto.display import display, Javascript
        self.background_color = None
        self.title = widgets.HTML("<b>%s</b>" % title)
        self.mouse_x = widgets.IntText()
        self.mouse_y = widgets.IntText()
        self.svg_canvas = widgets.HTML(
            self.get_html(onClick="window.clicked(evt, '%s', '%s')" % (self.mouse_x.model_id, self.mouse_y.model_id)))
        self.window = widgets.HBox([self.title,
                                    self.svg_canvas])
        display(Javascript("""
        window.clicked = function (evt, x_model, y_model) {
            var e = evt.srcElement.farthestViewportElement || evt.target;
            var dim = e.getBoundingClientRect();
            var x = evt.clientX - dim.left;
            var y = evt.clientY - dim.top;
            var manager = IPython.WidgetManager._managers[0];
            var model_prom = manager.get_model(x_model);
            model_prom.then(function(model) {
                model.set('value', Math.round(x));
                model.save_changes();
            });
            model_prom = manager.get_model(y_model);
            model_prom.then(function(model) {
                model.set('value', Math.round(y));
                model.save_changes();
            });
        };
        """))
        display(self.window)

    def close(self):
        self.window.close()

    def setBackground(self, color):
        self.background_color = color

    def draw(self, shape):
        super(GraphWin, self).draw(shape)
        self.svg_canvas.value = self.get_html(
            onClick="window.clicked(evt, '%s', '%s')" % (self.mouse_x.model_id, self.mouse_y.model_id))

    def _render(self):
        drawing = super(GraphWin, self)._render()
        if self.background_color:
            rect = _CRectangle((0,0), self.size)
            rect.canvas = self
            rect.matrix = []
            rect.fill(self.background_color)
            rect._add(drawing)
        return drawing

    def plot(self, x, y, color="black"):
        """
        Uses coordinant system.
        """
        p = Point(x, y)
        p.fill(color)
        p.draw(self)

    def plotPixel(self, x, y, color="black"):
        """
        Doesn't use coordinant system.
        """
        p = Point(x, y)
        p.fill(color)
        p.draw(self)
        p.t = lambda v: v
        p.tx = lambda v: v
        p.ty = lambda v: v

    def getMouse(self):
        """
        Waits for a mouse click.
        """
        # FIXME: this isn't working during an executing cell
        self.mouse_x.value = -1
        self.mouse_y.value = -1
        while self.mouse_x.value == -1 and self.mouse_y.value == -1:
            time.sleep(.1)
        return (self.mouse_x.value, self.mouse_y.value)

    def checkMouse(self):
        """
        Gets last click, or none.
        """
        return (self.mouse_x.value, self.mouse_y.value)

class Text(_CText):
    def __init__(self, center, text, **extras):
        super(Text, self).__init__(text,
                                     (center[0] - len(text) * 3.5,
                                      center[1] + 10), **extras)

class Rectangle(_CRectangle):
    def __init__(self, start, stop, **extras):
        super(Rectangle, self).__init__(
            (start[0],           start[1]),
            (stop[0] - start[0], stop[1] - start[1]),
            **extras)
        self.noFill()

class Oval(_CEllipse):
    """
    """
    def __init__(self, start, stop, **extras):
        super(Oval, self).__init__(
            (start[0] + (stop[0] + start[0])/2,
             start[1] + (stop[1] - start[1])/2),
            ((stop[0] - start[0])/2,
             (stop[1] - start[1])/2),
            **extras)
        self.noFill()

class Circle(_CCircle):
    """
    """
    def __init__(self, center, radius, **kwargs):
        super(Circle, self).__init__((center[0], center[1]),
                                     radius=radius, **kwargs)
        self.noFill()

class Line(_CLine):
    """
    """
    def __init__(self, *args, **kwargs):
        super(Line, self).__init__(*args, **kwargs)
        self.noFill()
