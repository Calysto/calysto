
__all__ = [
    # The container:
    'Canvas', 
    # Shapes:
    'Shape', 'Line', 'Circle', 'Text', 'Rectangle', 
    'Ellipse', 'Polyline', 'Polygon', 'Picture', 'Arc',
    'BarChart',
    # Pixel-based items:
    'Pixel', 'Color',
    # Units:
    'cm', 'em', 'ex', 'mm', 'pc', 'pt', 'px'
]

import svgwrite
from svgwrite import cm, em, ex, mm, pc, pt, px
import cairosvg
import numpy
import math
import copy
from cairosvg.parser import Tree

def rotate(x, y, length, radians):
    return (x + length * math.cos(-radians), y - length * math.sin(-radians))

class Canvas(object):
    def __init__(self, filename="noname.svg", size=(300, 300), **extras):
        if "debug" not in extras:
            extras["debug"] = False
        self.filename = filename
        self.size = size
        self.extras = extras
        self.shapes = []
        self._viewbox = None
        self.matrix = []
        self.fill_color = "purple"
        self.stroke_color = "black"
        self.stroke_width_width = 1
        self.fill_opacity = None
        self.stroke_opacity = None

    def __repr__(self):
        return "<Canvas %s>" % str(self.size)

    def _render(self):
        drawing = svgwrite.Drawing(self.filename, self.size, **self.extras)
        if self._viewbox:
            drawing.viewbox(*self._viewbox)
        return drawing

    def fill(self, color):
        self.fill_color = color
        if isinstance(color, Color):
            self.fill_opacity = color.alpha/255
        else:
            self.fill_opacity = None

    def noFill(self):
        self.fill_opacity = 0.0

    def stroke(self, color):
        self.stroke_color = color
        if isinstance(color, Color):
            self.stroke_opacity = color.alpha/255
        else:
            self.stroke_opacity = None

    def noStroke(self):
        self.stroke_opacity = 0.0

    def stroke_width(self, width):
        self.stroke_width_width = width

    def pushMatrix(self):
        self.matrix.append([])

    def translate(self, x, y):
        self.matrix[-1].append(("translate", x, y))

    def rotate(self, radians):
        self.matrix[-1].append(("rotate", radians * 180/math.pi))

    def popMatrix(self):
        self.matrix.pop()

    def viewbox(self, xmin, ymin, width, height):
        self._viewbox = (xmin, ymin, width, height)

    def save(self, filename=None):
        canvas = self._render()
        if filename:
            canvas.saveas(filename)
        else:
            canvas.save()

    def draw(self, shape):
        shape.canvas = self
        shape.matrix = copy.copy(self.matrix)
        if "fill" not in shape.extras:
            shape.extras["fill"] = self.fill_color
        if "stroke" not in shape.extras:
            shape.extras["stroke"] = self.stroke_color
        if "stroke-width" not in shape.extras:
            shape.extras["stroke-width"] = self.stroke_width_width
        if "stroke-opacity" not in shape.extras:
            if self.stroke_opacity is not None:
                shape.extras["stroke-opacity"] = self.stroke_opacity
        if "fill-opacity" not in shape.extras:
            if self.fill_opacity is not None:
                shape.extras["fill-opacity"] = self.fill_opacity
        self.shapes.append(shape)
        return self

    def undraw(self, shape):
        shape.canvas = None
        del self.shapes[shape]
        return self

    def _repr_svg_(self):
        canvas = self._render()
        for shape in self.shapes:
            shape._add(canvas)
        return canvas.tostring()

    def _repr_png_(self):
        return self.convert(format="png")

    def __str__(self):
        return self._repr_svg_()

    def convert(self, format="png", **kwargs):
        """
        png, ps, pdf, gif, jpg, svg
        returns image in format as string
        """
        surface = cairosvg.SURFACES[format.upper()]
        return surface.convert(bytestring=str(self), **kwargs)

    def toArray(self, dpi=96, **kwargs):
        """
        Converts svg-based image as a numpy array.
        """
        kwargs['bytestring'] = str(self)
        tree = Tree(**kwargs)
        pngsurface = cairosvg.SURFACES["PNG"](tree, None, dpi)
        buffer = pngsurface.cairo.get_data()
        width = pngsurface.cairo.get_width()
        height = pngsurface.cairo.get_height()
        array = numpy.frombuffer(buffer, numpy.uint8)
        array.shape = (width, height, 4) # 4 = rgb + alpha
        return array

    def getPixels(self):
        """
        Return a stream of pixels from current Canvas.
        """
        array = self.toArray()
        (width, height, depth) = array.size
        for x in range(width):
            for y in range(height):
                yield Pixel(array, x, y)

    def sortByZ(self):
        self.shapes.sort(key=lambda shape: shape.z)

class Pixel(object):
    """
    Wrapper interface to numpy array.
    """
    def __init__(self, array, x, y):
        self.array = array
        self.x = x
        self.y = y

    def getColor(self, x, y):
        return Color(*self.array[x, y])

    def setColor(self, x, y, color):
        self.array[x, y][0] = color.red
        self.array[x, y][1] = color.green
        self.array[x, y][2] = color.blue
        self.array[x, y][3] = color.alpha

    def getRGBA(self, x, y):
        return self.array[x, y]

class Color(object):
    def __init__(self, r, g=None, b=None, a=255):
        self.red = r
        if g is not None:
            self.green = g
        else:
            self.green = r
        if b is not None:
            self.blue = b
        else:
            self.blue = r
        self.alpha = a

    def __str__(self):
        def h(num):
            return ("0x%02x" % num)[-2:]
        return "#" + h(self.red) + h(self.green) + h(self.blue)

class Shape(object):
    def __init__(self, center=(0,0), **extras):
        if isinstance(center, tuple):
            self.center = list(center)
        else:
            self.center = center # use directly, no copy
        self.extras = extras
        self.canvas = None
        self.direction = 0
        self.pen = False
        self.z = 0

    def __repr__(self):
        return "<Shape %s>" % self.center

    def draw(self, canvas):
        canvas.draw(self)
        return canvas

    def _add(self, drawing):
        pass

    def _apply_matrices(self, shape, matrices):
        for matrix in matrices:
            for transform in matrix:
                self._apply_transform(shape, *transform)

    def _apply_transform(self, shape, transform, *args):
        if transform == "rotate":
            shape.rotate(*args)
        elif transform == "translate":
            shape.translate(*args)
        else:
            raise Exception("invalid transform: " + transform)

    def undraw(self, canvas):
        canvas.undraw(self)
        return canvas

    def forward(self, distance):
        start = self.center[:]
        self.center[0] += distance * math.cos(self.direction)
        self.center[1] += distance * math.sin(self.direction)
        if self.pen and self.canvas:
            Line(start, self.center, **self.extras).draw(self.canvas)
            return self.canvas

    def rotate(self, degrees): 
        self.direction -= (math.pi / 180.0) * degrees

    def fill(self, color): 
        self.extras["fill"] = str(color)
        if isinstance(color, Color):
            self.extras["fill-opacity"] = color.alpha/255
        else:
            self.extras["fill-opacity"] = 1.0

    def noFill(self):
        self.extras["fill-opacity"] = 0.0

    def noStroke(self):
        self.extras["stroke-opacity"] = 0.0

    def stroke(self, color): 
        self.extras["stroke"] = str(color)
        if isinstance(color, Color):
            self.extras["stroke-opacity"] = color.alpha/255
        else:
            self.extras["stroke-opacity"] = 1.0

    def stroke_width(self, width): 
        self.extras["stroke-width"] = width

    def moveToTop(self):
        self.canvas.shapes.remove(self)
        self.canvas.shapes.append(self)
        return self.canvas
        
    def moveToBottom(self):
        self.canvas.shapes.remove(self)
        self.canvas.shapes.insert(0, self)
        return self.canvas
        
class Circle(Shape):
    def __init__(self, center=(0,0), radius=1, **extras):
        super(Circle, self).__init__(center)
        self.radius = radius
        self.extras = extras

    def __repr__(self):
        return "<Circle %s, r=%s>" % (self.center, self.radius)

    def moveTo(self, center):
        self.center[:] = center # use directly, no copy
        return self.canvas

    def move(self, delta):
        self.center[:] = [self.center[0] + delta[0], self.center[1] + delta[1]]
        return self.canvas

    def _add(self, drawing):
        shape = drawing.circle(center=self.center, r=self.radius, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Arc(Shape):
    def __init__(self, center=(0,0), radius=1, start=0, stop=0, **extras):
        super(Arc, self).__init__(center)
        self.radius = radius
        self.start = start
        self.stop = stop
        self.extras = extras

    def __repr__(self):
        return "<Arc %s, r=%s>" % (self.center, self.radius)

    def moveTo(self, center):
        self.center[:] = center # use directly, no copy
        return self.canvas

    def move(self, delta):
        self.center[:] = [self.center[0] + delta[0], self.center[1] + delta[1]]
        return self.canvas

    def _add(self, drawing):
        current = self.start
        points = [(self.center[0], self.center[1])]
        while current < self.stop:
            points.append(rotate(self.center[0], self.center[1], self.radius, current))
            current += math.pi/180 * 5.0 # every five degrees
        extras = copy.copy(self.extras)
        extras["stroke-opacity"] = 0.0
        shape = drawing.polygon(points=points, **extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)
        extras = copy.copy(self.extras)
        extras["fill-opacity"] = 0.0
        shape = drawing.polyline(points=points[1:], **extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Line(Shape):
    def __init__(self, start=(0,0), end=(0,0), **extras):
        super(Line, self).__init__()
        if "stroke" not in extras:
            extras["stroke"] = "black"
        if "stroke-width" not in extras:
            extras["stroke-width"] = 1
        if isinstance(start, tuple):
            self.start = list(start)
        else:
            self.start = start # use directly, no copy
        if isinstance(end, tuple):
            self.end = list(end)
        else:
            self.end = end # use directly, no copy
        self.extras = extras

    def __repr__(self):
        return "<Line %s, %s>" % (self.start, self.end)

    def moveTo(self, start):
        diff_x = start[0] - self.start[0]
        diff_y = start[1] - self.start[1]
        self.start[:] = start
        self.end[:] = self.end[0] + diff_x, self.end[1] + diff_y
        return self.canvas

    def move(self, delta):
        self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        self.end[:] = self.end[0] + delta[0], self.end[1] + delta[1]
        return self.canvas

    def _add(self, drawing):
        shape = drawing.line(start=self.start, end=self.end, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Text(Shape):
    def __init__(self, text="", start=(0,0), **extras):
        super(Text, self).__init__()
        self.text = text
        if isinstance(start, tuple):
            self.start = list(start)
        else:
            self.start = start # use directly, no copy
        self.extras = extras

    def __repr__(self):
        return "<Text %s>" % self.start

    def moveTo(self, start):
        self.start[:] = start
        return self.canvas

    def move(self, delta):
        self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        return self.canvas

    def _add(self, drawing):
        shape = drawing.text(text=self.text, insert=self.start, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Rectangle(Shape):
    def __init__(self, start=(0,0), size=(1,1), rx=None, ry=None, **extras):
        super(Rectangle, self).__init__()
        if isinstance(start, tuple):
            self.start = list(start)
        else:
            self.start = start # use directly, no copy
        if isinstance(size, tuple):
            self.size = list(size)
        else:
            self.size = size # use directly, no copy
        self.rx = rx
        self.ry = ry
        self.extras = extras

    def __repr__(self):
        return "<Rectangle %s,%s>" % (self.start, self.size)

    def moveTo(self, start):
        self.start[:] = start
        return self.canvas

    def move(self, delta):
        self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        return self.canvas

    def _add(self, drawing):
        shape = drawing.rect(insert=self.start, size=self.size, rx=self.rx, ry=self.ry, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Ellipse(Shape):
    def __init__(self, center=(0,0), radii=(1,1), **extras):
        super(Ellipse, self).__init__(center)
        if isinstance(radii, tuple):
            self.radii = list(radii)
        else:
            self.radii = radii
        self.extras = extras

    def __repr__(self):
        return "<Ellipse %s>" % str(self.radii)

    def moveTo(self, center):
        self.center[:] = center # use directly, no copy
        return self.canvas

    def move(self, delta):
        self.center[:] = [self.center[0] + delta[0], self.center[1] + delta[1]]
        return self.canvas

    def _add(self, drawing):
        shape = drawing.ellipse(center=self.center, r=self.radii, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Polyline(Shape):
    def __init__(self, points=[], **extras):
        super(Polyline, self).__init__()
        self.points = points # not a copy FIXME
        self.extras = extras

    def __repr__(self):
        return "<Polyline %s>" % str(self.points)

    def moveTo(self, start):
        diff_x = start[0] - self.points[0][0]
        diff_y = start[1] - self.points[0][1]
        for i in range(len(self.points)):
            self.points[i] = self.points[i][0] + diff_x, self.points[i][1] + diff_y
        return self.canvas

    def move(self, delta):
        for i in range(len(self.points)):
            self.points[i] = self.points[i][0] + delta[0], self.points[i][1] + delta[1]
        return self.canvas

    def _add(self, drawing):
        shape = drawing.polyline(points=self.points, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Polygon(Shape):
    def __init__(self, points=[], **extras):
        super(Polygon, self).__init__()
        self.points = points # not a copy FIXME
        self.extras = extras

    def __repr__(self):
        return "<Polygon %s>" % str(self.points)

    def moveTo(self, start):
        diff_x = start[0] - self.points[0][0]
        diff_y = start[1] - self.points[0][1]
        for i in range(len(self.points)):
            self.points[i] = self.points[i][0] + diff_x, self.points[i][1] + diff_y
        return self.canvas

    def move(self, delta):
        for i in range(len(self.points)):
            self.points[i] = self.points[i][0] + delta[0], self.points[i][1] + delta[1]
        return self.canvas

    def _add(self, drawing):
        shape = drawing.polygon(points=self.points, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Picture(Shape):
    def __init__(self, href, start=None, size=None, **extras):
        super(Picture, self).__init__()
        self.href = href
        self.start = start
        self.size = size
        self.extras = extras

    def __repr__(self):
        return "<Picture %s,%s>" % (self.start, self.size)

    def moveTo(self, start):
        self.start[:] = start
        return self.canvas

    def move(self, delta):
        self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        return self.canvas

    def _add(self, drawing):
        shape = drawing.image(self.href, insert=self.start, size=self.size, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Plot(Canvas):
    """
    """
    def __init__(self, *args, **kwargs):
        super(Plot, self).__init__(*args, **kwargs)
        ## Borders:
        self.frame_size = 20
        self.x_offset = 20
        ## (upper left, width, height)
        self.plot_boundaries = [self.frame_size + self.x_offset, 
                                self.frame_size,
                                self.size[0] - 2 * self.frame_size - self.x_offset, 
                                self.size[1] - 2 * self.frame_size]
        self.plot_width = self.plot_boundaries[2] - self.plot_boundaries[0]
        self.plot_height = self.plot_boundaries[3] - self.plot_boundaries[1]
        self.plot_origin = (self.plot_boundaries[0], self.size[1] - self.frame_size * 2)
        self.background = Rectangle((self.plot_boundaries[0],
                                     self.plot_boundaries[1]),
                                    (self.plot_width, self.plot_height))
        self.background.noFill()
        self.background.stroke("#000000")
        self.draw(self.background)

class BarChart(Plot):
    """
    """
    def __init__(self, *args, **kwargs):
        super(BarChart, self).__init__(*args, **kwargs)
        self.data = kwargs.get("data", [])
        self.bar_width = self.plot_width / len(self.data)
        max_count = max(self.data)
        start_x = self.plot_origin[0] + self.bar_width * 0.125
        start_y = self.plot_origin[1]
        count = 0
        for item in self.data:
            self.draw( Rectangle((start_x + self.bar_width * count,
                                  start_y - item/max_count * self.plot_height), 
                                 (self.bar_width * 0.75, 
                                  item/max_count * self.plot_height)))
            count += 1
        # X legend:
        self.labels = kwargs.get("labels", [])
        count = 0
        for item in self.labels:
            self.draw( Text(item, (start_x + self.bar_width * count + self.bar_width/3,
                                   start_y + self.frame_size)))
            count += 1
        # Y legend:
        for count in range(4 + 1):
            self.draw( Text(str(max_count * count/4)[0:5],
                            (0, 1.2 * self.frame_size + self.plot_height - count/4 * self.plot_height)))

#g(**extras) # Group
#symbol(**extras)
#svg(insert=None, size=None, **extras)
#use(href, insert=None, size=None, **extras)
#a(href, target='_blank', **extras) # Link
#marker(insert=None, size=None, orient=None, **extras)
#script(href=None, content='', **extras)
#style(content='', **extras)
#linearGradient(start=None, end=None, inherit=None, **extras)
#radialGradient(center=None, r=None, focal=None, inherit=None, **extras)
#mask(start=None, size=None, **extras)
#clipPath(**extras)
#set(element=None, **extras)

#tspan(text, insert=None, x=[], y=[], dx=[], dy=[], rotate=[], **extras) # TextSpan
#tref(element, **extras)
#textPath(path, text, startOffset=None, method='align', spacing='exact', **extras)
#textArea(text=None, insert=None, size=None, **extras)
#path(d=None, **extras)

#animate(element=None, **extras)
#animateColor(element=None, **extras)
#animateMotion(element=None, **extras)
#animateTransform(transform, element=None, **extras)
#filter(start=None, size=None, resolution=None, inherit=None, **extras)
