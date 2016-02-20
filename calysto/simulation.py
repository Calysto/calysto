from calysto.graphics import (Canvas, Polygon, Rectangle, Circle,
                              Color, Line, Ellipse, Arc, Text)
import numpy as np
import traceback
import threading
import random
import math
import time
import sys
import os

SIMULATION = None

def rotateAround(x1, y1, length, angle):
    return Point(x1 + length * math.cos(-angle), y1 - length * math.sin(-angle))

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def pdistance(x1, y1, x2, y2, patches_size):
    pw = patches_size[0]
    ph = patches_size[1]
    min_x_diff = min(abs((x1 + pw) - x2), abs(x1 - x2), abs(x1 - (x2 + pw)))
    min_y_diff = min(abs((y1 + ph) - y2), abs(y1 - y2), abs(y1 - (y2 + ph)))
    return math.sqrt(min_x_diff ** 2 + min_y_diff ** 2)

class Point(object):
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return 2

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, pos):
        if pos == 0:
            return self.x
        elif pos == 1:
            return self.y
        elif pos == 2:
            return self.z

class Drawable(object):
    def __init__(self, points, color):
        self.points = points
        self.color = color

    def draw(self, canvas):
        polygon = Polygon([(p.x, p.y) for p in self.points])
        polygon.fill(self.color)
        polygon.noStroke()
        polygon.draw(canvas)

class Wall(Drawable):
    """
    Wall object in the simulated world.
    """

class Simulation(object):
    def __init__(self, w, h, *robots, **kwargs):
        global SIMULATION
        background_color = kwargs.get("background_color", None)
        draw_walls = kwargs.get("draw_walls", True)
        self.w = w
        self.h = h
        if background_color:
            self.background_color = background_color
        else:
            self.background_color = Color(0, 128, 0)
        self.scale = 250
        self.at_x = 0
        self.at_y = 0
        self.robots = []
        self.walls = []
        self.shapes = []
        if draw_walls:
            self.makeWall(0, 0, self.w, 10, Color(128, 0, 128))
            self.makeWall(0, 0, 10, self.h, Color(128, 0, 128))
            self.makeWall(0, self.h - 10.0, self.w, self.h, Color(128, 0, 128))
            self.makeWall(self.w - 10.0, 0.0, 10, self.h, Color(128, 0, 128))
        self.need_to_stop = threading.Event()
        self.is_running = threading.Event()
        self.brain_running = threading.Event()
        self.paused = threading.Event()
        self.clock = 0.0
        self.sim_time = .1  # every update advances this much time
        self.gui_time = .25 # update screen this often; used in watch
        self.gui_update = 5 # used in widget-based view
        for robot in robots:
            self.addRobot(robot)
        self.error = None
        SIMULATION = self

    def reset(self):
        self.clock = 0.0
        for robot in self.robots:
            robot.stop()
            robot.x = robot.ox
            robot.y = robot.oy
            robot.direction = robot.odirection
            if robot.brain:
                self.runBrain(robot.brain)

    def start_sim(self, gui=True, set_values={}, error=None):
        """
        Run the simulation in the background, showing the GUI by default.
        """
        self.error = error
        if not self.is_running.is_set():
            def loop():
                self.need_to_stop.clear()
                self.is_running.set()
                for robot in self.robots:
                    if robot.brain:
                        self.runBrain(robot.brain)
                count = 0
                while not self.need_to_stop.isSet():
                    if not self.paused.is_set():
                        self.clock += self.sim_time
                        for robot in self.robots:
                            try:
                                robot.update()
                            except Exception as exc:
                                self.need_to_stop.set()
                                if error:
                                    error.value = "Error: %s. Now stopping simulation." % str(exc)
                                else:
                                    raise
                    if gui:
                        self.draw()
                    if count % self.gui_update == 0:
                        if "canvas" in set_values:
                            set_values["canvas"].value = str(self.render())
                        if "energy" in set_values:
                            if len(self.robots) > 0:
                                set_values["energy"].value = str(self.robots[0].energy)
                    count += 1
                    self.realsleep(self.sim_time)
                    if self.robots[0].energy <= 0:
                        self.need_to_stop.set()
                self.is_running.clear()
                for robot in self.robots:
                    robot.stop()
            threading.Thread(target=loop).start()

    def render(self):
        canvas = Canvas(size=(self.w, self.h))
        rect = Rectangle((self.at_x, self.at_y), (self.w, self.h))
        rect.fill(self.background_color)
        rect.noStroke()
        rect.draw(canvas)
        for wall in self.walls:
            wall.draw(canvas)
        for shape in self.shapes:
            shape.draw(canvas)
        for robot in self.robots:
            robot.draw(canvas)
        if self.brain_running.is_set():
            if not self.paused.is_set():
                state = "Brain Running..."
            else:
                state = "Brain paused!"
        else:
            if not self.paused.is_set():
                state = "Idle"
            else:
                state = "Paused"
        clock = Text("%.1f %s" % (self.clock, state), (15, self.h - 15))
        clock.fill(Color(255, 255, 255))
        clock.stroke(Color(255, 255, 255))
        clock.stroke_width(1)
        clock.draw(canvas)
        return canvas

    def draw(self):
        """
        Render and draw the world and robots.
        """
        from calysto.display import display, clear_output
        canvas = self.render()
        clear_output(wait=True)
        display(canvas)

    def watch(self):
        """
        Watch a running simulation.
        """
        while True:
            self.draw()
            self.realsleep(self.gui_time)

    def stop_sim(self):
        self.need_to_stop.set()
        time.sleep(.250)
        for robot in self.robots:
            robot.stop()

    def realsleep(self, seconds):
        """
        Realtime sleep, to not overwhelm the system.
        """
        self.need_to_stop.wait(seconds)

    def sleep(self, seconds):
        """
        Sleep in simulated time.
        """
        start = self.time()
        while (self.time() - start < seconds and
               not self.need_to_stop.is_set()):
            self.need_to_stop.wait(self.sim_time)

    def time(self):
        return self.clock

    def runBrain(self, f):
        """
        Run a brain program in the background.
        """
        if self.error:
            self.error.value = ""
        def wrapper():
            self.brain_running.set()
            try:
                f()
            except KeyboardInterrupt:
                # Just stop
                pass
            except Exception as e:
                if self.error:
                    self.error.value = "<pre style='background: #fdd'>" + traceback.format_exc() + "</pre>"
                else:
                    raise
            finally:
                self.brain_running.clear()
            # Otherwise, will show error
        threading.Thread(target=wrapper).start()

    def makeWall(self, x, y, w, h, color):
        wall = Wall([Point(x, y),
                     Point(x + w, y),
                     Point(x + w, y + h),
                     Point(x, y + h)],
                    color)
        self.walls.append(wall)

    def set_gui_update(self, value):
        self.gui_update = value

    def setScale(self, s):
        ## scale the world... > 1 make it bigger
        self.scale = s * 250

    def addRobot(self, robot):
        self.robots.append(robot)
        robot.setSimulation(self)

class DiscreteSimulation(Simulation):
    def __init__(self, *args, **kwargs):
        super(DiscreteSimulation, self).__init__(*args, **kwargs)
        self.pwidth = kwargs.get("pwidth", 10)
        self.pheight = kwargs.get("pheight", 10)
        self.psize = (int(self.w/self.pwidth), int(self.h/self.pheight))
        self.items = {}
        self.items["f"] = self.drawFood
        self.initialize()
        self.gui_update = 1

    def initialize(self):
        self.patches = [[None for h in range(self.psize[1])] for w in range(self.psize[0])]

    def reset(self):
        for robot in self.robots:
            robot.stop()
            robot.x = robot.ox
            robot.y = robot.oy
            robot.direction = robot.odirection
            robot.energy = robot.oenergy
            robot.history = [robot.energy]

    def addCluster(self, cx, cy, item, count, lam_percent=.25):
        """
        Add a Poisson cluster of count items around (x,y).
        """
        dx, dy = map(lambda v: v * lam_percent, self.psize)
        total = 0
        while total < count:
            points = np.random.poisson(lam=(dx, dy), size=(count, 2))
            for x, y in points:
                px, py = (int(x - dx + cx), int(y - dy + cy))
                if self.getPatch(px, py) is None:
                    self.setPatch(px, py, item)
                    total += 1
                    if total == count:
                        break

    def setPatch(self, px, py, item):
        self.patches[int(px) % self.psize[0]][int(py) % self.psize[1]] = item

    def getPatch(self, px, py):
        return self.patches[px % self.psize[0]][py % self.psize[1]]

    def getPatchLocation(self, px, py):
        return (px % self.psize[0], py % self.psize[1])

    def render(self):
        self.shapes.clear()
        for x in range(0, int(self.w/self.pwidth)):
            for y in range(0, int(self.h/self.pheight)):
                if self.patches[x][y] and self.patches[x][y] in self.items:
                    self.items[self.patches[x][y]](x, y)
        return super(DiscreteSimulation, self).render()

    def drawFood(self, px, py):
        center = (px * self.pwidth + self.pwidth/2,
                  py * self.pheight + self.pheight/2)
        food = Circle(center, 5)
        food.fill("yellow")
        self.shapes.append(food)

    def drawWall(self, px, py):
        center = (px * self.pwidth + self.pwidth/2,
                  py * self.pheight + self.pheight/2)
        food = Circle(center, 5)
        food.fill("purple")
        self.shapes.append(food)

class Hit(object):
    def __init__(self, x, y, distance, color, start_x, start_y):
        self.x = x
        self.y = y
        self.distance = distance
        self.color = color
        self.start_x = start_x
        self.start_y = start_y

class Robot(object):
    def __init__(self, x, y, direction):
        """
        direction is in radians
        """
        self.simulation = None
        self.brain = None
        self.x = self.ox = x
        self.y = self.oy= y
        self.direction = self.odirection = direction
        self.debug = False
        self.vx = 0.0 ## velocity in x direction
        self.vy = 0.0 ## velocity in y direction
        self.va = 0.0 ## turn velocity
        ## sensors
        self.stalled = False
        self.bounding_box = [Point(0, 0)] * 4
        self.color = Color(255, 0, 0)
        self.ir_sensors = [None] * 2 # Hits
        self.max_ir = 1/5 # ratio of robot
        self.camera = [None] * 256 # Hits
        self.take_picture = threading.Event()
        self.picture_ready = threading.Event()
        self.body_points = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def setSimulation(self, simulation):
        self.simulation = simulation
        sx = [0.05, 0.05, 0.07, 0.07, 0.09, 0.09, 0.07,
              0.07, 0.05, 0.05, -0.05, -0.05, -0.07,
              -0.08, -0.09, -0.09, -0.08, -0.07, -0.05,
              -0.05]
        sy = [0.06, 0.08, 0.07, 0.06, 0.06, -0.06, -0.06,
              -0.07, -0.08, -0.06, -0.06, -0.08, -0.07,
              -0.06, -0.05, 0.05, 0.06, 0.07, 0.08, 0.06]
        self.body_points = []
        for i in range(len(sx)):
            self.body_points.append(Point(sx[i] * self.simulation.scale, sy[i] * self.simulation.scale))

    ### Continuous Movements:

    def sleep(self, seconds):
        self.simulation.sleep(seconds)
        if self.simulation.need_to_stop.is_set():
            raise KeyboardInterrupt()

    def forward(self, seconds, vx=5):
        """
        Move continuously in simulator for seconds and velocity vx.
        """
        self.vx = vx
        self.sleep(seconds)
        self.vx = 0

    def backward(self, seconds, vx=5):
        self.vx = -vx
        self.sleep(seconds)
        self.vx = 0

    def turnLeft(self, seconds, va=math.pi/180):
        self.va = va * 4
        self.sleep(seconds)
        self.va = 0

    def turnRight(self, seconds, va=math.pi/180):
        self.va = -va * 4
        self.sleep(seconds)
        self.va = 0

    def getIR(self, pos=None):
        ## 0 is on right, front
        ## 1 is on left, front
        if pos is None:
            return [self.getIR(0), self.getIR(1)]
        else:
            hit = self.ir_sensors[pos]
            if (hit is not None):
                return hit.distance / (self.max_ir * self.simulation.scale)
            else:
                return 1.0

    def takePicture(self):
        self.picture_ready.clear()
        self.take_picture.set()
        pic = None
        if self.picture_ready.wait(1.5):
            from PIL import Image
            pic = Image.new("RGB", (256, 128))
            size = max(self.simulation.w, self.simulation.h)
            for i in range(len(self.camera)):
                hit = self.camera[i]
                if (hit != None):
                    s = max(min(1.0 - hit.distance/size, 1.0), 0.0)
                    if isinstance(hit.color, Color):
                        r = hit.color.red
                        g = hit.color.green
                        b = hit.color.blue
                    else:
                        try:
                            import webcolors
                            r, g, b = webcolors.name_to_rgb(hit.color)
                        except:
                            r, g, b = (128, 128, 128)
                    hcolor = (int(r * s), int(g * s), int(b * s))
                    high = (1.0 - s) * 128
                    ##pg.line(i, 0 + high/2, i, 128 - high/2)
                else:
                    high = 0
                    hcolor = None
                for j in range(128):
                    if (j < high/2): ##256 - high/2.0): ## sky
                        pic.putpixel((i, j), (0, 0, 128))
                    elif (j < 128 - high/2): ##256 - high and hcolor != None): ## hit
                        if (hcolor != None):
                            pic.putpixel((i, j), hcolor)
                    else: ## ground
                        pic.putpixel((i, j), (0, 128, 0))
        if self.simulation.need_to_stop.is_set():
            raise KeyboardInterrupt()
        self.take_picture.clear()
        return pic

    def stop(self):
        self.vx = 0.0
        self.vy = 0.0
        self.va = 0.0

    def ccw(self, ax, ay, bx, by, cx, cy):
        ## counter clockwise
        return (((cy - ay) * (bx - ax)) > ((by - ay) * (cx - ax)))

    def intersect(self, ax, ay, bx, by, cx, cy, dx, dy):
        ## Return True if line segments AB and CD intersect
        return (self.ccw(ax, ay, cx, cy, dx, dy) != self.ccw(bx, by, cx, cy, dx, dy) and
                self.ccw(ax, ay, bx, by, cx, cy) != self.ccw(ax, ay, bx, by, dx, dy))

    def coefs(self, p1x, p1y, p2x, p2y):
        A = (p1y - p2y)
        B = (p2x - p1x)
        C = (p1x * p2y - p2x * p1y)
        return [A, B, -C]

    def intersect_coefs(self, L1_0, L1_1, L1_2, L2_0, L2_1, L2_2):
        D  = L1_0 * L2_1 - L1_1 * L2_0
        Dx = L1_2 * L2_1 - L1_1 * L2_2
        Dy = L1_0 * L2_2 - L1_2 * L2_0
        if (D != 0):
            x1 = Dx / D
            y1 = Dy / D
            return [x1, y1]
        else:
            return None

    def intersect_hit(self, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
        ## http:##stackoverflow.com/questions/20677795/find-the-point-of-intersecting-lines
        L1 = self.coefs(p1x, p1y, p2x, p2y)
        L2 = self.coefs(p3x, p3y, p4x, p4y)
        xy = self.intersect_coefs(L1[0], L1[1], L1[2], L2[0], L2[1], L2[2])
        ## now check to see on both segments:
        if (xy != None):
            lowx = min(p1x, p2x) - .1
            highx = max(p1x, p2x) + .1
            lowy = min(p1y, p2y) - .1
            highy = max(p1y, p2y) + .1
            if (lowx <= xy[0] and xy[0] <= highx and
                lowy <= xy[1] and xy[1] <= highy):
                lowx = min(p3x, p4x) - .1
                highx = max(p3x, p4x) + .1
                lowy = min(p3y, p4y) - .1
                highy = max(p3y, p4y) + .1
                if (lowx <= xy[0] and xy[0] <= highx and
                    lowy <= xy[1] and xy[1] <= highy):
                    return xy
        return None

    def castRay(self, x1, y1, a, maxRange):
        hits = []
        x2 = math.sin(a) * maxRange + x1
        y2 = math.cos(a) * maxRange + y1
        for wall in self.simulation.walls:
            ## if intersection, can't move
            v1, v2, v3, v4 = wall.points
            pos = self.intersect_hit(x1, y1, x2, y2, v1.x, v1.y, v2.x, v2.y)
            if (pos != None):
                dist = distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, wall.color, x1, y1))

            pos = self.intersect_hit(x1, y1, x2, y2, v2.x, v2.y, v3.x, v3.y)
            if (pos != None):
                dist = distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, wall.color, x1, y1))

            pos = self.intersect_hit(x1, y1, x2, y2, v3.x, v3.y, v4.x, v4.y)
            if (pos != None):
                dist = distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, wall.color, x1, y1))

            pos = self.intersect_hit(x1, y1, x2, y2, v4.x, v4.y, v1.x, v1.y)
            if (pos != None):
                dist = distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, wall.color, x1, y1))

        for robot in self.simulation.robots:
            if robot is self:
                continue
            v1, v2, v3, v4 = robot.bounding_box
            pos = self.intersect_hit(x1, y1, x2, y2, v1.x, v1.y, v2.x, v2.y)
            if (pos != None):
                dist = distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, robot.color, x1, y1))

            pos = self.intersect_hit(x1, y1, x2, y2, v2.x, v2.y, v3.x, v3.y)
            if (pos != None):
                dist = distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, robot.color, x1, y1))

            pos = self.intersect_hit(x1, y1, x2, y2, v3.x, v3.y, v4.x, v4.y)
            if (pos != None):
                dist = distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, robot.color, x1, y1))

            pos = self.intersect_hit(x1, y1, x2, y2, v4.x, v4.y, v1.x, v1.y)
            if (pos != None):
                dist = distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, robot.color, x1, y1))

        if len(hits) == 0:
            return None
        else:
            return self.min_hits(hits)

    def min_hits(self, hits):
        minimum = hits[0]
        for hit in hits:
            if (hit.distance < minimum.distance):
                minimum = hit
        return minimum

    def check_for_stall(self, px, py, pdirection):
        scale = self.simulation.scale
        p1 = rotateAround(px, py, 30/250.0 * scale, pdirection + math.pi/4 + 0 * math.pi/2)
        p2 = rotateAround(px, py, 30/250.0 * scale, pdirection + math.pi/4 + 1 * math.pi/2)
        p3 = rotateAround(px, py, 30/250.0 * scale, pdirection + math.pi/4 + 2 * math.pi/2)
        p4 = rotateAround(px, py, 30/250.0 * scale, pdirection + math.pi/4 + 3 * math.pi/2)
        for wall in self.simulation.walls:
            ## if intersection, can't move
            v1 = wall.points[0]
            v2 = wall.points[1]
            v3 = wall.points[2]
            v4 = wall.points[3]
            ## p1 to p2
            if (self.intersect(p1[0], p1[1], p2[0], p2[1],
                               v1.x, v1.y, v2.x, v2.y) or
                self.intersect(p1[0], p1[1], p2[0], p2[1],
                               v2.x, v2.y, v3.x, v3.y) or
                self.intersect(p1[0], p1[1], p2[0], p2[1],
                               v3.x, v3.y, v4.x, v4.y) or
                self.intersect(p1[0], p1[1], p2[0], p2[1],
                               v4.x, v4.y, v1.x, v1.y) or
                ## p2 to p3
                self.intersect(p2[0], p2[1], p3[0], p3[1],
                               v1.x, v1.y, v2.x, v2.y) or
                self.intersect(p2[0], p2[1], p3[0], p3[1],
                               v2.x, v2.y, v3.x, v3.y) or
                self.intersect(p2[0], p2[1], p3[0], p3[1],
                               v3.x, v3.y, v4.x, v4.y) or
                self.intersect(p2[0], p2[1], p3[0], p3[1],
                               v4.x, v4.y, v1.x, v1.y) or
                ## p3 to p4
                self.intersect(p3[0], p3[1], p4[0], p4[1],
                               v1.x, v1.y, v2.x, v2.y) or
                self.intersect(p3[0], p3[1], p4[0], p4[1],
                               v2.x, v2.y, v3.x, v3.y) or
                self.intersect(p3[0], p3[1], p4[0], p4[1],
                               v3.x, v3.y, v4.x, v4.y) or
                self.intersect(p3[0], p3[1], p4[0], p4[1],
                               v4.x, v4.y, v1.x, v1.y) or
                ## p4 to p1
                self.intersect(p4[0], p4[1], p1[0], p1[1],
                               v1.x, v1.y, v2.x, v2.y) or
                self.intersect(p4[0], p4[1], p1[0], p1[1],
                               v2.x, v2.y, v3.x, v3.y) or
                self.intersect(p4[0], p4[1], p1[0], p1[1],
                               v3.x, v3.y, v4.x, v4.y) or
                self.intersect(p4[0], p4[1], p1[0], p1[1],
                               v4.x, v4.y, v1.x, v1.y)):
                self.stalled = True
                break

        if not self.stalled:
            # keep checking for obstacles:
            for robot in self.simulation.robots:
                if robot is self:
                    continue
                v1, v2, v3, v4 = robot.bounding_box
                ## p1 to p2
                if (self.intersect(p1[0], p1[1], p2[0], p2[1],
                                   v1.x, v1.y, v2.x, v2.y) or
                    self.intersect(p1[0], p1[1], p2[0], p2[1],
                                   v2.x, v2.y, v3.x, v3.y) or
                    self.intersect(p1[0], p1[1], p2[0], p2[1],
                                   v3.x, v3.y, v4.x, v4.y) or
                    self.intersect(p1[0], p1[1], p2[0], p2[1],
                                   v4.x, v4.y, v1.x, v1.y) or
                    ## p2 to p3
                    self.intersect(p2[0], p2[1], p3[0], p3[1],
                                   v1.x, v1.y, v2.x, v2.y) or
                    self.intersect(p2[0], p2[1], p3[0], p3[1],
                                   v2.x, v2.y, v3.x, v3.y) or
                    self.intersect(p2[0], p2[1], p3[0], p3[1],
                                   v3.x, v3.y, v4.x, v4.y) or
                    self.intersect(p2[0], p2[1], p3[0], p3[1],
                                   v4.x, v4.y, v1.x, v1.y) or
                    ## p3 to p4
                    self.intersect(p3[0], p3[1], p4[0], p4[1],
                                   v1.x, v1.y, v2.x, v2.y) or
                    self.intersect(p3[0], p3[1], p4[0], p4[1],
                                   v2.x, v2.y, v3.x, v3.y) or
                    self.intersect(p3[0], p3[1], p4[0], p4[1],
                                   v3.x, v3.y, v4.x, v4.y) or
                    self.intersect(p3[0], p3[1], p4[0], p4[1],
                                   v4.x, v4.y, v1.x, v1.y) or
                    ## p4 to p1
                    self.intersect(p4[0], p4[1], p1[0], p1[1],
                                   v1.x, v1.y, v2.x, v2.y) or
                    self.intersect(p4[0], p4[1], p1[0], p1[1],
                                   v2.x, v2.y, v3.x, v3.y) or
                    self.intersect(p4[0], p4[1], p1[0], p1[1],
                                   v3.x, v3.y, v4.x, v4.y) or
                    self.intersect(p4[0], p4[1], p1[0], p1[1],
                                   v4.x, v4.y, v1.x, v1.y)):
                    self.stalled = True
                    break
        return (p1, p2, p3, p4)

    def bump_variability(self):
        return (random.random() * .2) - .1

    def update(self):
        scale = self.simulation.scale
        tvx = self.vx * math.sin(-self.direction + math.pi/2) + self.vy * math.cos(-self.direction + math.pi/2)
        tvy = self.vx * math.cos(-self.direction + math.pi/2) - self.vy * math.sin(-self.direction + math.pi/2)
        ## proposed positions:
        self.stalled = False
        if (self.vx != 0 or self.vy != 0 or self.va != 0):
            px = self.x + tvx/250.0 * scale
            py = self.y + tvy/250.0 * scale
            pdirection = self.direction - self.va
            pbox = self.check_for_stall(px, py, pdirection)
            if (not self.stalled):
                ## if no intersection, make move
                self.x = px
                self.y = py
                self.direction = pdirection
                self.bounding_box = pbox
            else:
                self.direction += self.bump_variability()

        ## update sensors, camera:
        ## on right:
        p = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction + math.pi/8)
        hit = self.castRay(p[0], p[1], -self.direction + math.pi/2.0, self.max_ir * scale)
        self.ir_sensors[0] = hit

        p = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction - math.pi/8)
        hit = self.castRay(p[0], p[1], -self.direction + math.pi/2, self.max_ir * scale)
        self.ir_sensors[1] = hit

        ## camera:
        if self.take_picture.is_set():
            for i in range(256):
                angle = i/256.0 * 60 - 30
                self.camera[i] = self.castRay(self.x, self.y, -self.direction + math.pi/2.0 - angle*math.pi/180.0, 1000)
            self.picture_ready.set()

    def draw(self, canvas):
        scale = self.simulation.scale
        if self.debug:
            ## bounding box:
            p1, p2, p3, p4 = self.bounding_box
            for line in [Line((p1[0], p1[1]), (p2[0], p2[1])),
                         Line((p2[0], p2[1]), (p3[0], p3[1])),
                         Line((p3[0], p3[1]), (p4[0], p4[1])),
                         Line((p4[0], p4[1]), (p1[0], p1[1]))]:
                line.stroke(Color(255, 255, 255))
                line.draw(canvas)

            for hit, offset in zip(self.ir_sensors, [math.pi/8, -math.pi/8]):
                if hit:
                    # FIXME: offset should be part ofsensor:
                    p = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction + offset)
                    # Draw hit:
                    ellipse = Ellipse((p[0], p[1]), (5, 5))
                    ellipse.fill(Color(0, 255, 0))
                    ellipse.draw(canvas)
                    ellipse = Ellipse((hit.x, hit.y), (5, 5))
                    ellipse.fill(Color(0, 255, 0))
                    ellipse.draw(canvas)

        self.draw_body(canvas)
        self.draw_sensors(canvas)

    def draw_body(self, canvas):
        scale = self.simulation.scale
        canvas.pushMatrix()
        canvas.translate(self.x, self.y)
        canvas.rotate(self.direction)
        ## body:
        if (self.stalled):
            canvas.fill(Color(128, 128, 128))
            canvas.stroke(Color(255, 255, 255))
        else:
            canvas.fill(self.color)
            canvas.noStroke()
        canvas.noStroke()
        polygon = Polygon(self.body_points)
        polygon.draw(canvas)
        ## Draw wheels:
        canvas.fill(Color(0))
        rect = Rectangle((-10/250.0 * scale, -23/250.0 * scale), (19/250.0 * scale, 5/250.0 * scale))
        rect.draw(canvas)
        rect = Rectangle((-10/250.0 * scale, 18/250.0 * scale), (19/250.0 * scale, 5/250.0 * scale))
        rect.draw(canvas)
        ## hole:
        canvas.fill(Color(0, 64, 0))
        ellipse = Ellipse((0, 0), (7/250.0 * scale, 7/250.0 * scale))
        ellipse.draw(canvas)
        ## fluke
        canvas.fill(Color(0, 64, 0))
        rect = Rectangle((15/250.0 * scale, -10/250.0 * scale), (4/250.0 * scale, 19/250.0 * scale))
        rect.draw(canvas)
        canvas.popMatrix()

    def draw_sensors(self, canvas):
        scale = self.simulation.scale
        ## draw sensors
        ## right front IR
        ## position of start of sensor:
        p1 = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction + math.pi/8)
        ## angle of sensor:
        p2 = rotateAround(p1[0], p1[1], self.getIR(0) * self.max_ir * scale, self.direction)
        dist = distance(p1[0], p1[1], p2[0], p2[1])
        if (self.getIR(0) < 1.0):
            canvas.stroke(Color(255))
            canvas.fill(Color(128, 0, 128, 64))
            arc = Arc((p1[0], p1[1]), dist, self.direction - .5, self.direction + .5)
            arc.draw(canvas)
        ## left front IR
        p1 = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction - math.pi/8)
        ## angle of sensor:
        p2 = rotateAround(p1[0], p1[1], self.getIR(1) * self.max_ir * scale, self.direction)
        dist = distance(p1[0], p1[1], p2[0], p2[1])
        if (self.getIR(1) < 1.0):
            canvas.stroke(Color(255))
            canvas.noStroke()
            canvas.fill(Color(128, 0, 128, 64))
            arc = Arc((p1[0], p1[1]), dist, self.direction - .5, self.direction + .5)
            arc.draw(canvas)

class DiscreteLadybug(Robot):
    def __init__(self, *args, **kwargs):
        super(DiscreteLadybug, self).__init__(*args, **kwargs)
        self.energy = kwargs.get("energy", 100)
        self.oenergy = self.energy
        self.history = [self.energy]
        self.block_types = ["b", "w"] ## bugs and walls, block movement
        self.edible = {"f": 20}
        self.state = "0"
        self.rules = None

    def draw_body(self, canvas):
        px, py = self.x, self.y
        center = (px * self.simulation.pwidth + self.simulation.pwidth/2,
                  py * self.simulation.pheight + self.simulation.pheight/2)
        ladybug = Circle(center, self.simulation.pwidth)
        ladybug.fill("red")
        ladybug.draw(canvas)
        head = Arc(center, self.simulation.pwidth,  self.direction - math.pi/2, self.direction + math.pi/2)
        head.fill("black")
        head.draw(canvas)

    def forward(self, distance):
        self.move(distance, 0)

    def backward(self, distance):
        self.move(-distance, 0)

    def set_energy(self, energy):
        self.energy = energy
        self.history.append(self.energy)

    def turnLeft(self, angle):
        self.direction -= angle * math.pi/180
        self.set_energy(self.energy - angle/360 * 4.0)
        self.direction = self.direction % (math.pi * 2.0)

    def stop(self):
        #self.energy -= 0.75
        pass

    def turnRight(self, angle):
        self.direction += angle * math.pi/180
        # 90 degree == 1 unit
        self.set_energy(self.energy - angle/360 * 4.0)
        self.direction = self.direction % (math.pi * 2.0)

    def sign(self, value):
        if value == 0:
            return 0
        elif value < 0:
            return -1
        elif value > 0:
            return 1

    def move(self, tx, ty):
        for step in range(int(max(abs(tx), abs(ty)))):
            dx = self.sign(tx)
            dy = self.sign(ty)
            x = dx * math.sin(-self.direction + math.pi/2) + dy * math.cos(-self.direction + math.pi/2)
            y = dx * math.cos(-self.direction + math.pi/2) - dy * math.sin(-self.direction + math.pi/2)
            # check to see if move is possible:
            px, py = self.simulation.getPatchLocation(int(self.x + x), int(self.y + y))
            # update energy (even if agent was unable to move):
            # distance of 1 is 1 unit of energy:
            self.set_energy(self.energy - pdistance(self.x, self.y, px, py, self.simulation.psize))
            spot = self.simulation.patches[px][py]
            # if can move, make move:
            if spot is None or spot not in self.block_types:
                # if food, eat it:
                if spot in self.edible:
                    self.set_energy(self.energy + self.edible[spot])
                # move into:
                self.simulation.patches[px][py] = 'b'
                # Move out of:
                self.simulation.setPatch(int(self.x), int(self.y), None)
                # Update location:
                self.x, self.y = px, py

    def parseRule(self, srule):
        parts = []
        args = []
        current = ""
        state = "begin"
        for s in srule:
            if state == "begin":
                if s == "#" and current == "" and len(parts) == 0:
                    return None
                if s in [" ", "\n", "\t"]: # whitespace
                    if current:
                        parts.append(current)
                        current = ""
                elif s == "(": # args
                    state = "args"
                    if current:
                        parts.append(current)
                        current = ""
                else:
                    current += s
            elif state == "args":
                if s in [" ", "\n", "\t"]: # whitespace
                    if current:
                        args.append(current)
                        current = ""
                elif s == ")": # args
                    state = "begin"
                    if current:
                        args.append(current)
                        current = ""
                elif s == ",": # args
                    if current:
                        args.append(current)
                        current = ""
                else:
                    current += s
        if current:
            parts.append(current)
        parts.insert(-1, args)
        # state, match, "->", action, args, state
        if len(parts) == 1 and len(args) == 0:
            return None
        elif len(parts) != 6:
            raise Exception("Invalid length of rule in '%s'" % srule)
        elif parts[2] != "->":
            raise Exception("Item #3 should be => in '%s'" % srule)
        return parts

    def applyAction(self, command, args):
        if command == "turnLeft":
            if args[0] not in ["90", "180"]:
                raise Exception("Invalid angle: must be 90 or 180")
            self.turnLeft(float(args[0]))
        elif command == "turnRight":
            if args[0] not in ["90", "180"]:
                raise Exception("Invalid angle: must be 90 or 180")
            self.turnRight(float(args[0]))
        elif command == "forward":
            if not (1 <= float(args[0]) <= 9):
                raise Exception("Invalid distance: must be >= 1 or <= 9")
            self.forward(float(args[0]))
        elif command == "backward":
            if not (1 <= float(args[0]) <= 9):
                raise Exception("Invalid distance: must be >= 1 or <= 9")
            self.backward(float(args[0]))
        elif command == "stop":
            self.stop()

    def update(self):
        if self.rules is None:
            self.set_energy(self.energy - 0.75) # cost of being alive
            return
        firedRule = False
        rules = self.rules.strip()
        rules = rules.split("\n")
        if len(rules) == 0:
            raise Exception("Need at least one rule")
        ox, oy = self.x, self.y
        for rule in rules:
            # state, match, "->", action, args, state
            # match fff, *f*, f**, **f, no rule match, no movement
            parts = self.parseRule(rule)
            if parts:
                state, match, arrow, action, args, next_state = parts
                sense = self.getSenses()
                if self.state == state and self.match(match, sense):
                    self.applyAction(action, args)
                    self.state = next_state
                    firedRule = True
                    break
        if (ox, oy) == (self.x, self.y):
            self.set_energy(self.energy - 0.75) # cost of being alive
        if not firedRule:
            raise Exception("No rule matched")

    def getSenses(self):
        senses = []
        #self.positions = []
        # dx: forward/backward; dy: left/right


        for dx,dy in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]:
        #for dx,dy in [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1]]:
            x = dx * math.sin(-self.direction + math.pi/2) + dy * math.cos(-self.direction + math.pi/2)
            y = dx * math.cos(-self.direction + math.pi/2) - dy * math.sin(-self.direction + math.pi/2)
            item = self.simulation.getPatch(round(self.x + x), round(self.y + y))
            senses.append(item)
            #self.positions.append((round(self.x + x), round(self.y + y)))
        return senses

    def match(self, rule, world):
        if len(rule) != len(world):
            raise Exception("Matching part requires 5 characters: '%s'" % rule)
        for w,r in zip(world, rule):
            if w != r and r != "*":
                return False
        return True

class LadyBug(Robot):
    def draw_body(self, canvas):
        scale = self.simulation.scale
        canvas.pushMatrix()
        canvas.translate(self.x, self.y)
        canvas.rotate(self.direction)
        # Draw with front to right
        width  = 0.15 * scale
        length = 0.2 * scale
        for x in [-length/3, 0, length/3]:
            for side in [-1, 1]:
                end = x + (random.random() * length/5) - length/10
                leg = Line((x, width/3 * side), (end, (width/3 + width/4) * side))
                leg.stroke_width(3)
                leg.draw(canvas)
                leg = Line((end, (width/3 + width/4) * side), (end - length/5, (width/3 + width/4 + length/5) * side))
                leg.stroke_width(3)
                leg.draw(canvas)
        body = Ellipse((0,0), (length/2, width/2))
        if not self.stalled:
            body.fill(self.color)
        else:
            body.fill(Color(128, 128, 128))
        body.draw(canvas)
        head = Arc((width/2, 0), width/3, -math.pi/2, math.pi/2)
        head.fill(Color(0, 0, 0))
        head.draw(canvas)
        line = Line((width/1.5, 0), (-width/1.5, 0))
        line.draw(canvas)
        for x,y in [(length/5, width/5), (0, width/10),
                    (-length/5, -width/3), (length/5, -width/5),
                    (-length/4, width/4)]:
            spot = Ellipse((x, y), (length/20, length/20))
            spot.fill(Color(0, 0, 0))
            spot.draw(canvas)
        eye = Ellipse((length/2, width/5), (.01 * scale, .01 * scale))
        eye.fill(Color(255, 255, 255))
        eye.draw(canvas)
        eye = Ellipse((length/2, -width/5), (.01 * scale, .01 * scale))
        eye.fill(Color(255, 255, 255))
        eye.draw(canvas)
        canvas.popMatrix()

class Spider(Robot):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_ir = 1/2 # ratio of robot

    def bump_variability(self):
        return 0.0

    def draw_body(self, canvas):
        scale = self.simulation.scale
        canvas.pushMatrix()
        canvas.translate(self.x, self.y)
        canvas.rotate(self.direction)
        # Draw with front to right
        width  = 0.2 * scale
        length = 0.2 * scale
        for x in [-length/8 * 2.5, -length/8, length/8, length/8 * 2.5]:
            for side in [-1, 1]:
                end = x + (random.random() * length/5) - length/10
                leg = Line((x + length/7, 0), (end, (width/4 + width/4) * side))
                leg.stroke_width(5)
                leg.stroke(Color(0, 0, 0))
                leg.draw(canvas)
                leg = Line((end, (width/4 + width/4) * side), (end - length/5, (width/4 + width/4 + length/5) * side))
                leg.stroke_width(3)
                leg.stroke(Color(0, 0, 0))
                leg.draw(canvas)
        body = Ellipse((0,0), (length/3, width/3))
        head = Circle((width/2, 0), width/5)
        if not self.stalled:
            body.fill(Color(0, 0, 0))
            head.fill(Color(0, 0, 0))
        else:
            body.fill(Color(128, 128, 128))
            head.fill(Color(128, 128, 128))
        body.draw(canvas)
        head.draw(canvas)
        eye = Ellipse((length/2, width/5), (.01 * scale, .01 * scale))
        eye.fill(Color(255, 255, 255))
        eye.draw(canvas)
        eye = Ellipse((length/2, -width/5), (.01 * scale, .01 * scale))
        eye.fill(Color(255, 255, 255))
        eye.draw(canvas)
        canvas.popMatrix()

### ------------------------------------

def get_sim():
    return SIMULATION

def get_robot(index=0):
    return SIMULATION.robots[index]

def loadSimulation(sim_filename):
    sim_folder, filename = os.path.split(__file__)
    sim_folder = os.path.join(sim_folder, "simulations")
    if sim_folder not in sys.path:
        sys.path.append(sim_folder)
    mod = __import__(sim_filename)
    return mod.makeSimulation()

class DiscreteView(object):
    def __init__(self, sim_filename):
        from ipywidgets import widgets
        self.sim_filename = sim_filename
        self.canvas = widgets.HTML()
        self.go_button = widgets.Button(description="Go")
        self.stop_button = widgets.Button(description="Stop")
        self.step_button = widgets.Button(description="Step")
        self.reset_button = widgets.Button(description="Restart")
        self.error = widgets.HTML("")
        self.simulation = loadSimulation(self.sim_filename)
        self.simulation.sim_time = 0.2
        self.canvas.value = str(self.simulation.render())
        self.energy_widget = widgets.Text(str(self.simulation.robots[0].energy))
        #self.energy_widget.disabled = True
        self.sim_widgets = widgets.VBox([self.canvas,
                                    widgets.HBox([self.go_button,
                                                  self.stop_button,
                                                  self.step_button,
                                                  self.reset_button,
                                                  self.energy_widget,
                                                  ]),
                                    self.error])
        self.go_button.on_click(self.go)
        self.stop_button.on_click(self.stop)
        self.step_button.on_click(self.step)
        self.reset_button.on_click(self.reset)

    def go(self, obj):
        self.simulation.start_sim(gui=False, set_values={"canvas": self.canvas,
                                                         "energy": self.energy_widget},
                                  error=self.error)

    def stop(self, obj=None):
        self.simulation.stop_sim()

    def step(self, obj=None):
        self.simulation.clock += self.simulation.sim_time
        for robot in self.simulation.robots:
            robot.update()
        self.canvas.value = str(self.simulation.render())
        self.energy_widget.value = str(self.simulation.robots[0].energy)

    def reset(self, obj=None):
        self.simulation.clock = 0.0
        self.simulation.stop_sim()
        self.simulation.initialize()
        self.canvas.value = str(self.simulation.render())
        self.energy_widget.value = str(self.simulation.robots[0].energy)

    def setRobot(self, pos, robot):
        self.simulation.robots[pos] = robot

    def addCluster(self, x, y, item, count):
        self.simulation.addCluster(x, y, item, count)

    def render(self):
        return self.simulation.render()

    clock = property(lambda self: self.simulation.clock)

def View(sim_filename):
    try:
        from ipywidgets import widgets
    except:
        from IPython.html import widgets

    def stop_sim(obj):
        simulation.stop_sim()
        pause_button.visible = False

    def restart(x, y, direction):
        simulation.stop_sim()
        time.sleep(.250)
        simulation.reset()
        canvas.value = str(simulation.render())
        simulation.start_sim(gui=False, set_values={"canvas": canvas}, error=error)
        pause_button.visible = True

    def stop_and_start(obj):
        simulation.stop_sim()
        time.sleep(.250)
        simulation.start_sim(gui=False, set_values={"canvas": canvas}, error=error)
        pause_button.visible = True

    def toggle_pause(obj):
        if simulation.paused.is_set():
            simulation.paused.clear()
            pause_button.description = "Pause Simulation"
        else:
            simulation.paused.set()
            pause_button.description = "Resume Simulation"

    canvas = widgets.HTML()
    #stop_button = widgets.Button(description="Stop Brain")
    stop_sim_button = widgets.Button(description="Stop Simulation")
    restart_button = widgets.Button(description="Restart Simulation")
    pause_button = widgets.Button(description="Pause Simulation")
    gui_button = widgets.IntSlider(description="GUI Update Interval", min=1, max=10, value=5)
    error = widgets.HTML("")

    simulation = loadSimulation(sim_filename)
    simulation.start_sim(gui=False, set_values={"canvas": canvas}, error=error)
    #simulation.stop_sim()

    canvas.value = str(simulation.render())

    sim_widgets = widgets.VBox([canvas,
                                gui_button,
                                widgets.HBox([stop_sim_button,
                                              #stop_button,
                                              restart_button,
                                              pause_button,
                                              ]),
                                error])


    stop_button.on_click(stop_and_start)
    stop_sim_button.on_click(stop_sim)
    restart_button.on_click(lambda obj: restart(550, 350, -math.pi/2))
    pause_button.on_click(toggle_pause)
    gui_button.on_trait_change(lambda *args: simulation.set_gui_update(gui_button.value), "value")

    return sim_widgets

class DNARobot(object):
    def __init__(self, robot, dna=None):
        from calysto.ai import conx
        self.clen = 3
        self.net = conx.SRN(verbosity=-1)
        self.net.addSRNLayers(5, 3, 1)
        self.dna_length = len(self.net.arrayify() * self.clen)
        self.robot = robot
        if dna is None:
            self.dna = self.make_dna(self.dna_length)
        else:
            self.dna = dna
        self.net.unArrayify(self.make_array_from_dna(self.dna))
        self.net["context"].setActivations([.25, .25, .25])

    def codon2weight(self, codon):
        """
        Turn a codon of "000" to "999" to a number between
        -5.0 and 5.0.
        """
        length = len(codon)
        retval = int(codon)
        return retval/(10 ** (length - 1)) - 5.0

    def weight2codon(self, weight, length=None):
        """
        Given a weight between -5 and 5, turn it into
        a codon, eg "000" to "999"
        """
        if length is None:
            length = self.clen
        retval = 0
        weight = min(max(weight + 5.0, 0), 10.0) * (10 ** (length - 1))
        for i in range(length):
            if i == length - 1: # last one
                d = int(round(weight / (10 ** (length - i - 1))))
            else:
                d = int(weight / (10 ** (length - i - 1)))
            weight = weight % (10 ** (length - i - 1))
            retval += d * (10 ** (length - i - 1))
        return ("%0" + str(length) + "d") % retval

    def make_dna(self, length=None):
        import random
        if length is None:
            length = self.dna_length
        return "".join([random.choice("0123456789") for i in range(length)])

    def make_array_from_dna(self, dna):
        array = []
        for i in range(0, len(dna), self.clen):
            codon = dna[i:i+self.clen]
            array.append(self.codon2weight(codon))
        return array

    def draw(self, canvas):
        self.robot.draw(canvas)

    def stop(self):
        self.robot.stop()

    def update(self):
        def sense2num(s):
            if s == 'w':
                return 0.75
            elif s == None:
                return 0.5
            else:
                return 0.25
        senses = self.robot.getSenses()
        fsenses = list(map(sense2num, senses))
        v = self.net.propagate(input=fsenses)
        self.net.postBackprop()
        if v[0] < .33:
            self.robot.turnLeft(90)
        elif v[0] < .66:
            self.robot.forward(1)
        else:
            self.robot.turnRight(90)

    x = property(lambda self: self.robot.x,
                lambda self, value: setattr(self.robot, "x", value))
    y = property(lambda self: self.robot.y,
                lambda self, value: setattr(self.robot, "y", value))
    direction = property(lambda self: self.robot.direction,
                        lambda self, value: setattr(self.robot, "direction", value))
    energy = property(lambda self: self.robot.energy,
                      lambda self, value: setattr(self.robot, "energy", value))
    ox = property(lambda self: self.robot.ox)
    oy = property(lambda self: self.robot.oy)
    odirection = property(lambda self: self.robot.odirection)
    oenergy = property(lambda self: self.robot.oenergy)
    history = property(lambda self: self.robot.history,
                      lambda self, value: setattr(self.robot, "history", value))
