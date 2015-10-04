from calysto.display import display, clear_output
from calysto.graphics import (Canvas, Polygon, Rectangle, Circle,
                              Color, Line, Ellipse, Arc, Text)
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
    def __init__(self, w, h, *robots):
        global SIMULATION
        self.w = w
        self.h = h
        self.scale = 250
        self.at_x = 0
        self.at_y = 0
        self.robots = []
        self.walls = []
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
        self.gui_time = .25 # update screen this often
        for robot in robots:
            self.addRobot(robot)
        self.error = None
        SIMULATION = self

    def reset(self):
        for robot in self.robots:
            robot.stop()
            robot.x = robot.ox
            robot.y = robot.oy
            robot.direction = robot.odirection
            if robot.brain:
                self.runBrain(robot.brain)

    def start_sim(self, gui=True, set_value=None, error=None):
        """
        Run the simulation in the background, showing the GUI by default.
        """
        self.error = error
        self.clock = 0.0
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
                            robot.update()
                    if gui:
                        self.draw()
                    if set_value and count % 2 == 0:
                        set_value.value = str(self.render())
                    count += 1
                    self.realsleep(self.sim_time)
                self.is_running.clear()
                for robot in self.robots:
                    robot.stop()
            threading.Thread(target=loop).start()

    def render(self):
        canvas = Canvas(size=(self.w, self.h))
        rect = Rectangle((self.at_x, self.at_y), (self.w, self.h))
        rect.fill(Color(0, 128, 0))
        rect.noStroke()
        rect.draw(canvas)
        for wall in self.walls:
            wall.draw(canvas)
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
        clock.fill("white")
        clock.stroke("white")
        clock.stroke_width(1)
        clock.draw(canvas)
        return canvas

    def draw(self):
        """
        Render and draw the world and robots.
        """
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
                    self.error.value = "<pre style='background: #fdd>" + traceback.format_exc() + "</pre>"
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

    def setScale(self, s):
        ## scale the world... > 1 make it bigger
        self.scale = s * 250
      
    def addRobot(self, robot):
        self.robots.append(robot)
        robot.setSimulation(self)

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

    ### Movements:

    def sleep(self, seconds):
        self.simulation.sleep(seconds)
        if self.simulation.need_to_stop.is_set():
            raise KeyboardInterrupt()

    def forward(self, seconds, vx=5):
        self.vx = vx
        self.sleep(seconds)
        self.vx = 0
    
    def backward(self, seconds, vx=5):
        self.vx = -vx
        self.sleep(seconds)
        self.vx = 0
    
    def turnLeft(self, seconds, va=math.pi/180):
        self.va = va * 2
        self.sleep(seconds)
        self.va = 0
    
    def turnRight(self, seconds, va=math.pi/180):
        self.va = -va * 2
        self.sleep(seconds)
        self.va = 0
    
    def getIR(self, pos=None):
        ## 0 is on right, front
        ## 1 is on left, front
        if pos is None:
            return [self.getIR(0), self.getIR(1)]
        elif (self.ir_sensors[pos] != None):
            return self.ir_sensors[pos].distance / (self.max_ir * self.simulation.scale)
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
                    r = hit.color.red
                    g = hit.color.green
                    b = hit.color.blue
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
    
    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
     
    def castRay(self, x1, y1, a, maxRange):
        hits = []
        x2 = math.sin(a) * maxRange + x1
        y2 = math.cos(a) * maxRange + y1
        for wall in self.simulation.walls:
            ## if intersection, can't move
            v1, v2, v3, v4 = wall.points
            pos = self.intersect_hit(x1, y1, x2, y2, v1.x, v1.y, v2.x, v2.y)
            if (pos != None):
                dist = self.distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, wall.color, x1, y1))
            
            pos = self.intersect_hit(x1, y1, x2, y2, v2.x, v2.y, v3.x, v3.y)
            if (pos != None):
                dist = self.distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, wall.color, x1, y1))
            
            pos = self.intersect_hit(x1, y1, x2, y2, v3.x, v3.y, v4.x, v4.y)
            if (pos != None):
                dist = self.distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, wall.color, x1, y1))
            
            pos = self.intersect_hit(x1, y1, x2, y2, v4.x, v4.y, v1.x, v1.y)
            if (pos != None):
                dist = self.distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, wall.color, x1, y1))

        for robot in self.simulation.robots:
            if robot is self:
                continue
            v1, v2, v3, v4 = robot.bounding_box
            pos = self.intersect_hit(x1, y1, x2, y2, v1.x, v1.y, v2.x, v2.y)
            if (pos != None):
                dist = self.distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, robot.color, x1, y1))
            
            pos = self.intersect_hit(x1, y1, x2, y2, v2.x, v2.y, v3.x, v3.y)
            if (pos != None):
                dist = self.distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, robot.color, x1, y1))
            
            pos = self.intersect_hit(x1, y1, x2, y2, v3.x, v3.y, v4.x, v4.y)
            if (pos != None):
                dist = self.distance(pos[0], pos[1], x1, y1)
                hits.append(Hit(pos[0], pos[1], dist, robot.color, x1, y1))
            
            pos = self.intersect_hit(x1, y1, x2, y2, v4.x, v4.y, v1.x, v1.y)
            if (pos != None):
                dist = self.distance(pos[0], pos[1], x1, y1)
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
        dist = self.distance(p1[0], p1[1], p2[0], p2[1])
        if (self.getIR(0) < 1.0):
            canvas.stroke(Color(255))
            canvas.fill(Color(128, 0, 128, 64))
            arc = Arc((p1[0], p1[1]), dist, self.direction - .5, self.direction + .5)
            arc.draw(canvas)
        ## left front IR
        p1 = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction - math.pi/8)
        ## angle of sensor:
        p2 = rotateAround(p1[0], p1[1], self.getIR(1) * self.max_ir * scale, self.direction)
        dist = self.distance(p1[0], p1[1], p2[0], p2[1])
        if (self.getIR(1) < 1.0):
            canvas.stroke(Color(255))
            canvas.noStroke()
            canvas.fill(Color(128, 0, 128, 64))
            arc = Arc((p1[0], p1[1]), dist, self.direction - .5, self.direction + .5)
            arc.draw(canvas)

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
            body.fill("gray")
        body.draw(canvas)
        head = Arc((width/2, 0), width/3, -math.pi/2, math.pi/2)
        head.fill("black")
        head.draw(canvas)
        line = Line((width/1.5, 0), (-width/1.5, 0))
        line.draw(canvas)
        for x,y in [(length/5, width/5), (0, width/10), 
                    (-length/5, -width/3), (length/5, -width/5),
                    (-length/4, width/4)]:
            spot = Ellipse((x, y), (length/20, length/20))
            spot.fill("black")
            spot.draw(canvas)
        eye = Ellipse((length/2, width/5), (.01 * scale, .01 * scale))
        eye.fill("white")
        eye.draw(canvas)
        eye = Ellipse((length/2, -width/5), (.01 * scale, .01 * scale))
        eye.fill("white")
        eye.draw(canvas)
        canvas.popMatrix()

class Spider(Robot):
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
                leg.stroke("black")
                leg.draw(canvas)
                leg = Line((end, (width/4 + width/4) * side), (end - length/5, (width/4 + width/4 + length/5) * side))
                leg.stroke_width(3)
                leg.stroke("black")
                leg.draw(canvas)
        body = Ellipse((0,0), (length/3, width/3))
        head = Circle((width/2, 0), width/5)
        if not self.stalled:
            body.fill("black")
            head.fill("black")
        else:
            body.fill("gray")
            head.fill("gray")
        body.draw(canvas)
        head.draw(canvas)
        eye = Ellipse((length/2, width/5), (.01 * scale, .01 * scale))
        eye.fill("white")
        eye.draw(canvas)
        eye = Ellipse((length/2, -width/5), (.01 * scale, .01 * scale))
        eye.fill("white")
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

def View(sim_filename):
    try:
        from ipywidgets import widgets
    except:
        from IPython.html import widgets

    def restart(x, y, direction):
        simulation.stop_sim()
        time.sleep(.250)
        simulation.reset()
        canvas.value = str(simulation.render())
        simulation.start_sim(gui=False, set_value=canvas, error=error)

    def stop_and_start(obj):
        simulation.stop_sim()
        time.sleep(.250)
        simulation.start_sim(gui=False, set_value=canvas, error=error)

    def toggle_pause(obj):
        if simulation.paused.is_set():
            simulation.paused.clear()
            pause_button.description = "Pause Simulation"
        else:
            simulation.paused.set()
            pause_button.description = "Resume Simulation"

    canvas = widgets.HTML()
    stop_button = widgets.Button(description="Stop Brain")
    stop_sim_button = widgets.Button(description="Stop Simulation")
    restart_button = widgets.Button(description="Restart Simulation")
    pause_button = widgets.Button(description="Pause Simulation")
    error = widgets.HTML("")

    simulation = loadSimulation(sim_filename)
    simulation.start_sim(gui=False, set_value=canvas, error=error)

    canvas.value = str(simulation.render())

    sim_widgets = widgets.VBox([canvas, 
                                widgets.HBox([stop_sim_button, 
                                              stop_button, 
                                              restart_button,
                                              pause_button]),
                                error])

    stop_button.on_click(stop_and_start)
    stop_sim_button.on_click(lambda obj: simulation.stop_sim())
    restart_button.on_click(lambda obj: restart(550, 350, -math.pi/2))
    pause_button.on_click(toggle_pause)

    return sim_widgets
