
__all__ = [
    # The container:
    'Canvas', 
    # Shapes:
    'Shape', 'Line', 'Circle', 'Text', 'Rectangle', 
    'Ellipse', 'Polyline', 'Polygon', 'Picture',
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
from cairosvg.parser import Tree

class Canvas(object):
    def __init__(self, filename="noname.svg", size=(300, 300), **extras):
        if "debug" not in extras:
            extras["debug"] = False
        self.filename = filename
        self.size = size
        self.extras = extras
        self.shapes = []
        self._viewbox = None

    def __repr__(self):
        return "<Canvas %s>" % str(self.size)

    def _render(self):
        canvas = svgwrite.Drawing(self.filename, self.size, **self.extras)
        if self._viewbox:
            canvas.viewbox(*self._viewbox)
        return canvas

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
    def __init__(self, r, g, b, a=255):
        self.red = r
        self.green = g
        self.blue = b
        self.alpha = a

    def __str__(self):
        def h(num):
            return ("0x%02x" % num)[-2:]
        return "#" + h(self.red) + h(self.green) + h(self.blue)

class Shape(object):
    def __init__(self, center=(0,0), **extra):
        if isinstance(center, tuple):
            self.center = list(center)
        else:
            self.center = center # use directly, no copy
        if "fill" not in extra:
            extra["fill"] = "purple"
        if "stroke" not in extra:
            extra["stroke"] = "black"
        if "stroke_width" not in extra:
            extra["stroke_width"] = 1
        self.extra = extra
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

    def undraw(self, canvas):
        canvas.undraw(self)
        return canvas

    def forward(self, distance):
        start = self.center[:]
        self.center[0] += distance * math.cos(self.direction)
        self.center[1] += distance * math.sin(self.direction)
        if self.pen and self.canvas:
            Line(start, self.center, **self.extra).draw(self.canvas)
            return self.canvas

    def rotate(self, degrees): 
        self.direction -= (math.pi / 180.0) * degrees

    def fill(self, color): 
        self.extra["fill"] = str(color)
        if isinstance(color, Color):
            self.extra["fill-opacity"] = color.aplha/255
        elif "fill-opacity" in self.extra:
            del self.extra["fill-opacity"] 

    def noFill(self):
        self.extra["fill-opacity"] = 0.0

    def noStroke(self):
        self.extra["stroke-opacity"] = 0.0

    def stroke(self, color): 
        self.extra["stroke"] = str(color)
        if isinstance(color, Color):
            self.extra["stroke-opacity"] = color.aplha/255
        elif "stroke-opacity" in self.extra:
            del self.extra["stroke-opacity"] 

    def stroke_width(self, width): 
        self.extra["stroke_width"] = width

    def moveToTop(self):
        self.canvas.shapes.remove(self)
        self.canvas.shapes.append(self)
        return self.canvas
        
    def moveToBottom(self):
        self.canvas.shapes.remove(self)
        self.canvas.shapes.insert(0, self)
        return self.canvas
        

class Circle(Shape):
    def __init__(self, center=(0,0), radius=1, **extra):
        super(Circle, self).__init__(center)
        if "fill" not in extra:
            extra["fill"] = "purple"
        if "stroke" not in extra:
            extra["stroke"] = "black"
        if "stroke_width" not in extra:
            extra["stroke_width"] = 1
        self.radius = radius
        self.extra = extra

    def __repr__(self):
        return "<Circle %s, r=%s>" % (self.center, self.radius)

    def moveTo(self, center):
        self.center[:] = center # use directly, no copy
        return self.canvas

    def move(self, delta):
        self.center[:] = [self.center[0] + delta[0], self.center[1] + delta[1]]
        return self.canvas

    def _add(self, drawing):
        drawing.add(drawing.circle(center=self.center, r=self.radius, **self.extra))

class Line(Shape):
    def __init__(self, start=(0,0), end=(0,0), **extra):
        super(Line, self).__init__()
        if "stroke" not in extra:
            extra["stroke"] = "black"
        if "stroke_width" not in extra:
            extra["stroke_width"] = 1
        if isinstance(start, tuple):
            self.start = list(start)
        else:
            self.start = start # use directly, no copy
        if isinstance(end, tuple):
            self.end = list(end)
        else:
            self.end = end # use directly, no copy
        self.extra = extra

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
        drawing.add(drawing.line(start=self.start, end=self.end, **self.extra))

class Text(Shape):
    def __init__(self, text="", start=(0,0), **extra):
        super(Text, self).__init__()
        self.text = text
        if isinstance(start, tuple):
            self.start = list(start)
        else:
            self.start = start # use directly, no copy
        self.extra = extra

    def __repr__(self):
        return "<Text %s>" % self.start

    def moveTo(self, start):
        self.start[:] = start
        return self.canvas

    def move(self, delta):
        self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        return self.canvas

    def _add(self, drawing):
        drawing.add(drawing.text(text=self.text, insert=self.start, **self.extra))

class Rectangle(Shape):
    def __init__(self, start=(0,0), size=(1,1), rx=None, ry=None, **extra):
        super(Rectangle, self).__init__()
        if "fill" not in extra:
            extra["fill"] = "purple"
        if "stroke" not in extra:
            extra["stroke"] = "black"
        if "stroke_width" not in extra:
            extra["stroke_width"] = 1
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
        self.extra = extra

    def __repr__(self):
        return "<Rectangle %s,%s>" % (self.start, self.size)

    def moveTo(self, start):
        self.start[:] = start
        return self.canvas

    def move(self, delta):
        self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        return self.canvas

    def _add(self, drawing):
        drawing.add(drawing.rect(insert=self.start, size=self.size, rx=self.rx, ry=self.ry, **self.extra))

class Ellipse(Shape):
    def __init__(self, center=(0,0), radii=(1,1), **extra):
        super(Ellipse, self).__init__(center)
        if "fill" not in extra:
            extra["fill"] = "purple"
        if "stroke" not in extra:
            extra["stroke"] = "black"
        if "stroke_width" not in extra:
            extra["stroke_width"] = 1
        if isinstance(radii, tuple):
            self.radii = list(radii)
        else:
            self.radii = radii
        self.extra = extra

    def __repr__(self):
        return "<Ellipse %s>" % str(self.radii)

    def moveTo(self, center):
        self.center[:] = center # use directly, no copy
        return self.canvas

    def move(self, delta):
        self.center[:] = [self.center[0] + delta[0], self.center[1] + delta[1]]
        return self.canvas

    def _add(self, drawing):
        drawing.add(drawing.ellipse(center=self.center, r=self.radii, **self.extra))

class Polyline(Shape):
    def __init__(self, points=[], **extra):
        super(Polyline, self).__init__()
        self.points = points # not a copy FIXME
        self.extra = extra

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
        drawing.add(drawing.polyline(points=self.points, **self.extra))

class Polygon(Shape):
    def __init__(self, points=[], **extra):
        super(Polygon, self).__init__()
        if "fill" not in extra:
            extra["fill"] = "purple"
        if "stroke" not in extra:
            extra["stroke"] = "black"
        if "stroke_width" not in extra:
            extra["stroke_width"] = 1
        self.points = points # not a copy FIXME
        self.extra = extra

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
        drawing.add(drawing.polygon(points=self.points, **self.extra))

class Picture(Shape):
    def __init__(self, href, start=None, size=None, **extra):
        super(Picture, self).__init__()
        self.href = href
        self.start = start
        self.size = size
        self.extra = extra

    def __repr__(self):
        return "<Picture %s,%s>" % (self.start, self.size)

    def moveTo(self, start):
        self.start[:] = start
        return self.canvas

    def move(self, delta):
        self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        return self.canvas

    def _add(self, drawing):
        drawing.add(drawing.image(self.href, insert=self.start, size=self.size, **self.extra))

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

#g(**extra) # Group
#symbol(**extra)
#svg(insert=None, size=None, **extra)
#use(href, insert=None, size=None, **extra)
#a(href, target='_blank', **extra) # Link
#marker(insert=None, size=None, orient=None, **extra)
#script(href=None, content='', **extra)
#style(content='', **extra)
#linearGradient(start=None, end=None, inherit=None, **extra)
#radialGradient(center=None, r=None, focal=None, inherit=None, **extra)
#mask(start=None, size=None, **extra)
#clipPath(**extra)
#set(element=None, **extra)

#tspan(text, insert=None, x=[], y=[], dx=[], dy=[], rotate=[], **extra) # TextSpan
#tref(element, **extra)
#textPath(path, text, startOffset=None, method='align', spacing='exact', **extra)
#textArea(text=None, insert=None, size=None, **extra)
#path(d=None, **extra)

#animate(element=None, **extra)
#animateColor(element=None, **extra)
#animateMotion(element=None, **extra)
#animateTransform(transform, element=None, **extra)
#filter(start=None, size=None, resolution=None, inherit=None, **extra)
