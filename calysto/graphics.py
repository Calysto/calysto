"""
Parts based on John Zelle's graphics.py
http://mcsp.wartburg.edu/zelle/python/ppics2/index.html

LICENSE: This is open-source software released under the terms of the
GPL (http://www.gnu.org/licenses/gpl.html).
"""

__all__ = [
    # The container:
    'Canvas',
    # Shapes:
    'Shape', 'Line', 'Circle', 'Text', 'Rectangle',
    'Ellipse', 'Polyline', 'Polygon', 'Picture', 'Arc',
    'BarChart', 'Point', 'Turtle',
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
import io
from cairosvg.parser import Tree

def rotate(x, y, length, radians):
    return (x + length * math.cos(-radians), y - length * math.sin(-radians))

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

class Transform:
    """
    Internal class for 2-D coordinate transformations
    """
    def __init__(self, w, h, xlow, ylow, xhigh, yhigh):
        # w, h are width and height of window
        # (xlow,ylow) coordinates of lower-left [raw (0,h-1)]
        # (xhigh,yhigh) coordinates of upper-right [raw (w-1,0)]
        xspan = (xhigh-xlow)
        yspan = (yhigh-ylow)
        self.xbase = xlow
        self.ybase = yhigh
        self.xscale = xspan/float(w-1)
        self.yscale = yspan/float(h-1)

    def screen(self,x,y):
        # Returns x,y in screen (actually window) coordinates
        xs = (x-self.xbase) / self.xscale
        ys = (self.ybase-y) / self.yscale
        return int(xs+0.5),int(ys+0.5)

    def world(self,xs,ys):
        # Returns xs,ys in world coordinates
        x = xs*self.xscale + self.xbase
        y = self.ybase - ys*self.yscale
        return x,y

class Canvas(object):
    def __init__(self, size=(300, 300), filename="noname.svg", **extras):
        if "debug" not in extras:
            extras["debug"] = False
        self.filename = filename
        self.size = size
        self.extras = extras
        self.shapes = []
        self._viewbox = None
        self.matrix = []
        self.fill_color = Color(128, 0, 128)
        self.stroke_color = Color(0, 0, 0)
        self.stroke_width_width = 1
        self.fill_opacity = None
        self.stroke_opacity = None
        self.trans = None

    def __repr__(self):
        return "<Canvas %s>" % str(self.size)

    def _render(self, **attribs):
        drawing = svgwrite.Drawing(self.filename, self.size, **self.extras)
        if self._viewbox:
            drawing.viewbox(*self._viewbox)
        for key in attribs:
            drawing.attribs[key] = attribs[key]
        for shape in self.shapes:
            shape._add(drawing)
        return drawing

    def setCoords(self, x1, y1, x2, y2):
        """Set coordinates of window to run from (x1,y1) in the
        lower-left corner to (x2,y2) in the upper-right corner."""
        self.trans = Transform(self.size[0], self.size[1], x1, y1, x2, y2)

    def toScreen(self, x, y):
        if self.trans:
            return self.trans.screen(x,y)
        else:
            return x,y

    def toWorld(self, x, y):
        if self.trans:
            return self.trans.world(x,y)
        else:
            return x,y

    def toScaleX(self, v):
        if self.trans:
            return v/self.trans.xscale
        else:
            return v

    def toScaleY(self, v):
        if self.trans:
            return v/self.trans.yscale
        else:
            return v

    def fill(self, color):
        self.fill_color = color
        if isinstance(color, Color):
            self.fill_opacity = color.alpha/255
        else:
            self.fill_opacity = None

    def setFill(self, color):
        self.fill(color)

    def setOutline(self, color):
        self.stroke(color)

    def setWidth(self, pixels):
        self.stroke_width(pixels)

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

    def scale(self, x, y):
        self.matrix[-1].append(("scale", x, y))

    def popMatrix(self):
        self.matrix.pop()

    def viewbox(self, xmin, ymin, width, height):
        self._viewbox = (xmin, ymin, width, height)

    def save(self, filename=None, **attribs):
        format = "svg"
        if filename and "." in filename:
            format = filename.rsplit(".", 1)[1]
        if format == "svg":
            drawing = self._render(**attribs)
            if filename:
                drawing.saveas(filename)
            else:
                drawing.save()
        else:
            im = self.toPIL(**attribs)
            im.save(filename, format=format)

    def draw(self, shape):
        shape.canvas = self
        shape.t = shape.canvas.toScreen
        shape.tx = shape.canvas.toScaleX
        shape.ty = shape.canvas.toScaleY
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
        del self.shapes[self.shapes.index(shape)]
        return self

    def get_html(self, **attribs):
        if "onClick" in attribs:
            onClick = attribs["onClick"]
            del attribs["onClick"]
        return self._repr_svg_(**attribs)

    def _repr_svg_(self, **attribs):
        drawing = self._render(**attribs)
        return drawing.tostring()

    def _repr_png_(self, **attribs):
        return self.convert(format="png", **attribs)

    def __str__(self):
        return self._repr_svg_()

    def convert(self, format="png", **kwargs):
        """
        png, ps, pdf, gif, jpg, svg
        returns image in format as bytes
        """
        surface = cairosvg.SURFACES[format.upper()]
        return surface.convert(bytestring=str(self), **kwargs)

    def toPIL(self, **attribs):
        """
        Convert canvas to a PIL image
        """
        import PIL.Image

        png = self._repr_png_(**attribs)
        sfile = io.BytesIO(png)
        pil = PIL.Image.open(sfile)
        background = PIL.Image.new('RGBA', pil.size, (255, 255, 255))
        # Paste the image on top of the background
        background.paste(pil, pil)
        im = background.convert('RGB').convert('P', palette=PIL.Image.ADAPTIVE)
        return im

    def toGIF(self, **attribs):
        """
        Convert canvas to GIF bytes
        """
        im = self.toPIL(**attribs)
        sfile = io.BytesIO()
        im.save(sfile, format="gif")
        return sfile.getvalue()

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

    def start_movie(self):
        self.frames = []

    def frame(self):
        self.frames.append(self.toGIF())

    def save_movie(self):
        pass

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

    def clear(self):
        """
        Clear all of the shapes.
        """
        self.shapes.clear()

class Pixel(object):
    """
    Wrapper interface to numpy array.
    """
    def __init__(self, array, x, y):
        self.array = array
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

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
        # Transforms:
        self.t = self.tx = self.ty = None
        self.direction = 0
        self.pen = False
        self.z = 0

    def __repr__(self):
        return "<Shape %s>" % self.center

    def draw(self, canvas):
        canvas.draw(self)
        return canvas

    def clone(self):
        """
        """
        pass
        #return clone, not drawn

    def _add(self, drawing):
        # Shape._add
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

    def undraw(self):
        self.canvas.undraw(self)
        self.canvas = None
        self.t = self.tx = self.ty = None

    def forward(self, distance):
        start = self.center[:]
        self.center[0] += distance * math.cos(self.direction)
        self.center[1] += distance * math.sin(self.direction)
        if self.pen and self.canvas:
            Line(start, self.center, **self.extras).draw(self.canvas)
            return self.canvas

    def turn(self, degrees):
        """
        Turn the shape direction by these degrees.
        """
        self.direction -= (math.pi / 180.0) * degrees

    def fill(self, color):
        self.extras["fill"] = str(color)
        if isinstance(color, Color):
            self.extras["fill-opacity"] = color.alpha/255
        else:
            self.extras["fill-opacity"] = 1.0

    def setFill(self, color):
        self.fill(color)

    def setOutline(self, color):
        self.stroke(color)

    def setWidth(self, pixels):
        self.stroke_width(pixels)

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

    def getP1(self):
        """
        Left, upper point
        """
        return Point(self.center[0] - self.radius,
                     self.center[1] - self.radius)

    def getP2(self):
        """
        Right, lower point
        """
        return Point(self.center[0] + self.radius,
                     self.center[1] + self.radius)

    def __repr__(self):
        return "<Circle %s, r=%s>" % (self.center, self.radius)

    def moveTo(self, center):
        self.center[:] = center # use directly, no copy
        return self.canvas

    def move(self, delta, delta_y=None):
        if delta_y is None:
            self.center[:] = [self.center[0] + delta[0], self.center[1] + delta[1]]
        else:
            self.center[:] = [self.center[0] + delta, self.center[1] + delta_y]
        return self.canvas

    def _add(self, drawing):
        if self.canvas.trans:
            shape = drawing.ellipse(center=self.t(*self.center),
                                    r=(self.tx(self.radius),
                                       self.ty(self.radius)),
                                    **self.extras)
        else:
            shape = drawing.circle(center=self.center, r=self.radius, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Point(Circle):
    def __init__(self, x, y, **extras):
        super(Point, self).__init__((x, y), 1)

    def __repr__(self):
        return "<Point (%s,%s)>" % (self.center[0], self.center[1])

    def _add(self, drawing):
        if self.canvas.trans:
            shape = drawing.circle(center=self.t(*self.center), r=self.radius, **self.extras)
        else:
            shape = drawing.circle(center=self.center, r=self.radius, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

    def __getitem__(self, pos):
        return self.center[pos]

    def __iter__(self):
        yield self.center[0]
        yield self.center[1]

    def getX(self):
        return self.center[0]

    def getY(self):
        return self.center[1]

class Arc(Shape):
    def __init__(self, center=(0,0), radius=1, start=0, stop=0, **extras):
        super(Arc, self).__init__(center)
        self.radius = radius
        self.start = start
        self.stop = stop
        self.extras = extras

    def getP1(self):
        """
        Left, upper point
        """
        return Point(self.center[0] - self.radius,
                     self.center[1] - self.radius)

    def getP2(self):
        """
        Right, lower point
        """
        return Point(self.center[0] + self.radius,
                     self.center[1] + self.radius)

    def __repr__(self):
        return "<Arc %s, r=%s>" % (self.center, self.radius)

    def moveTo(self, center):
        self.center[:] = center # use directly, no copy
        return self.canvas

    def move(self, delta, delta_y=None):
        if delta_y is None:
            self.center[:] = [self.center[0] + delta[0], self.center[1] + delta[1]]
        else:
            self.center[:] = [self.center[0] + delta, self.center[1] + delta_y]
        return self.canvas

    def _add(self, drawing):
        current = self.start
        if self.canvas.trans:
            points = [self.t(*self.center)]
        else:
            points = [(self.center[0], self.center[1])]
        while current < self.stop:
            if self.canvas.trans:
                c = self.t(*self.center)
                # FIXME: allow scale in x and y dimensions:
                points.append(rotate(c[0], c[1], self.tx(self.radius), current))
            else:
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
            extras["stroke"] = Color(0, 0, 0)
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

    def move(self, delta, delta_y=None):
        if delta_y is None:
            self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
            self.end[:] = self.end[0] + delta[0], self.end[1] + delta[1]
        else:
            self.start[:] = self.start[0] + delta, self.start[1] + delta_y
            self.end[:] = self.end[0] + delta, self.end[1] + delta_y
        return self.canvas

    def _add(self, drawing):
        if self.canvas.trans:
            shape = drawing.line(start=self.t(*self.start),
                                 end=self.t(*self.end), **self.extras)
        else:
            shape = drawing.line(start=self.start, end=self.end, **self.extras)
        self._apply_matrices(shape, self.matrix)
        drawing.add(shape)

class Turtle(object):
    def __init__(self, canvas, center, angle=0):
        self.canvas = canvas
        self.x = center[0]
        self.y = center[1]
        self.angle = angle * math.pi/180.0
        self.angle_units = "degrees"
        self.pen = "down"
        self.fill_color = Color(128, 0, 128)
        self.stroke = Color(0, 0, 0)
        self.stroke_width = 1
        self.fill_opacity = None
        self.stroke_opacity = None
        self.arrow = None
        self.draw_arrow()


    def draw_arrow(self):
        if self.arrow:
            self.arrow.undraw()
        self.canvas.pushMatrix()
        self.arrow = Polygon([(-10,   5),
                              (  0,   0),
                              (-10,  -5),
                              ( -5,   0)])
        self.canvas.translate(self.x, self.y)
        self.canvas.rotate(self.angle)
        self.arrow.draw(self.canvas)
        self.canvas.popMatrix()

    def forward(self, distance):
        new_x, new_y = rotate(self.x, self.y, distance, self.angle)
        if self.pen == "down":
            line = Line((self.x, self.y), (new_x, new_y),
                        **{"stroke": self.stroke,
                           "stroke-width": self.stroke_width})
            line.draw(self.canvas)
        self.x, self.y = new_x, new_y
        self.draw_arrow()
        return self.canvas

    def backward(self, distance):
        return self.forward(-distance)

    def left(self, angle):
        return self.right(-angle)

    def right(self, angle):
        if self.angle_units == "degrees":
            angle = angle * math.pi/180
        self.angle += angle
        self.angle = self.angle % (math.pi * 2)
        self.draw_arrow()
        return self.canvas

    def goto(self, new_x, new_y):
        if self.pen == "down":
            line = Line((self.x, self.y), (new_x, new_y),
                        **{"stroke": self.stroke,
                           "stroke-width": self.stroke_width})
            line.draw(self.canvas)
        self.x, self.y = new_x, new_y
        self.draw_arrow()
        return self.canvas

    def penup(self):
        self.pen = "up"

    def pendown(self):
        self.pen = "down"

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

    def move(self, delta, delta_y=None):
        if delta_y is None:
            self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        else:
            self.start[:] = self.start[0] + delta, self.start[1] + delta_y
        return self.canvas

    def _add(self, drawing):
        if self.canvas.trans:
            shape = drawing.text(text=self.text, insert=self.t(*self.start), **self.extras)
        else:
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

    def move(self, delta, delta_y=None):
        if delta_y is None:
            self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        else:
            self.start[:] = self.start[0] + delta, self.start[1] + delta_y
        return self.canvas

    def _add(self, drawing):
        if self.canvas.trans:
            shape = drawing.rect(insert=self.t(*self.start),
                                 size=(self.tx(self.size[0]), self.ty(self.size[1])),
                                 rx=self.rx,
                                 ry=self.ry,
                                 **self.extras)
        else:
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

    def move(self, delta, delta_y=None):
        if delta_y is None:
            self.center[:] = [self.center[0] + delta[0], self.center[1] + delta[1]]
        else:
            self.center[:] = [self.center[0] + delta, self.center[1] + delta_y]
        return self.canvas

    def _add(self, drawing):
        if self.canvas.trans:
            shape = drawing.ellipse(center=self.t(*self.center),
                                    r=(self.tx(self.radii[0]),
                                       self.ty(self.radii[1])),
                                    **self.extras)
        else:
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

    def move(self, delta, delta_y=None):
        for i in range(len(self.points)):
            if delta_y is None:
                self.points[i] = self.points[i][0] + delta[0], self.points[i][1] + delta[1]
            else:
                self.points[i] = self.points[i][0] + delta, self.points[i][1] + delta_y
        return self.canvas

    def _add(self, drawing):
        if self.canvas.trans:
            shape = drawing.polyline(points=map(lambda p: self.t(*p), self.points), **self.extras)
        else:
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

    def move(self, delta, delta_y=None):
        for i in range(len(self.points)):
            if delta_y is None:
                self.points[i] = self.points[i][0] + delta[0], self.points[i][1] + delta[1]
            else:
                self.points[i] = self.points[i][0] + delta, self.points[i][1] + delta_y
        return self.canvas

    def _add(self, drawing):
        if self.canvas.trans:
            shape = drawing.polygon(points=map(lambda p: self.t(*p), self.points), **self.extras)
        else:
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

    def move(self, delta, delta_y=None):
        if delta_y is None:
            self.start[:] = self.start[0] + delta[0], self.start[1] + delta[1]
        else:
            self.start[:] = self.start[0] + delta, self.start[1] + delta_y
        return self.canvas

    def _add(self, drawing):
        if self.canvas.trans:
            shape = drawing.image(self.href,
                                  insert=self.t(*self.start),
                                  size=self.t(*self.size), **self.extras)
        else:
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
        self.background.stroke(Color(0, 0, 0))
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
