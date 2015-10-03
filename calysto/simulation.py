from calysto.display import display, clear_output
from calysto.graphics import (Canvas, Polygon, Rectangle, 
                              Color, Line, Ellipse, Arc)
import threading
import math
import time

def rotateAround(x1, y1, length, angle):
    return [x1 + length * math.cos(-angle), y1 - length * math.sin(-angle)]

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

class World(object):
    def __init__(self, w, h):
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
        self.clock = 0.0
        self.sim_time = .1  # every update advances this much time
        self.gui_time = .25 # update screen this often

    def start_sim(self, gui=True):
        """
        Run the simulation in the background, showing the GUI by default.
        """
        if not self.is_running.is_set():
            def loop():
                self.need_to_stop.clear()
                self.is_running.set()
                while not self.need_to_stop.isSet():
                    for robot in self.robots:
                        robot.update()
                    if gui:
                        self.draw()
                    self.realsleep(self.sim_time)
                    self.clock += self.sim_time
                self.is_running.clear()
            threading.Thread(target=loop).start()

    def draw(self):
        """
        Render and draw the world and robots.
        """
        canvas = Canvas(size=(self.w, self.h))
        rect = Rectangle((self.at_x, self.at_y), (self.w, self.h))
        rect.fill(Color(0, 128, 0))
        rect.noStroke()
        rect.draw(canvas)
        for wall in self.walls:
            wall.draw(canvas)
        for robot in self.robots:
            robot.draw(canvas)
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
        def wrapper():
            try:
                f()
            except KeyboardInterrupt:
                # Just stop
                pass
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
      robot.world = self

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
        self.world = None
        self.x = x
        self.y = y
        self.direction = direction
        self.debug = False
        self.vx = 0.0 ## velocity in x direction
        self.vy = 0.0 ## velocity in y direction
        self.va = 0.0 ## turn velocity
        ## sensors 
        self.stalled = False
        self.bounding_box = [0] * 4
        self.robot_color = Color(255, 0, 0)
        self.ir_sensors = [None] * 2
        self.max_ir = 1/5 # ratio of robot
        self.camera = [0] * 256

    def sleep(self, seconds):
        self.world.sleep(seconds)
        if self.world.need_to_stop.is_set():
            raise KeyboardInterrupt()

    def forward(self, seconds, vx=5):
        self.vx = vx
        self.sleep(seconds)
        self.vx = 0
        if self.world.need_to_stop.is_set():
            raise KeyboardInterrupt()
    
    def backward(self, seconds, vx=5):
        self.vx = -vx
        self.sleep(seconds)
        self.vx = 0
        if self.world.need_to_stop.is_set():
            raise KeyboardInterrupt()
    
    def turnLeft(self, seconds, va=math.pi/180):
        self.va = va
        self.sleep(seconds)
        self.va = 0
        if self.world.need_to_stop.is_set():
            raise KeyboardInterrupt()
    
    def turnRight(self, seconds, va=math.pi/180):
        self.va = -va
        self.sleep(seconds)
        self.va = 0
        if self.world.need_to_stop.is_set():
            raise KeyboardInterrupt()
    
    def getIR(self, pos):
        ## 0 is on right, front
        ## 1 is on left, front
        if (self.ir_sensors[pos] != None):
            return self.ir_sensors[pos].distance / (self.max_ir * self.world.scale)
        else:
            return 1.0
    
    def takePicture(self):
        pic = PImage(256, 128)
        size = max(self.world.w, self.world.h)
        for i in range(len(self.camera)):
            hit = self.camera[i]
            if (hit != None):
                s = max(min(1.0 - hit.distance/size, 1.0), 0.0)
                r = red(hit.col)
                g = green(hit.col)
                b = blue(hit.col)
                hcolor = color(r * s, g * s, b * s)
                high = (1.0 - s) * 128
                ##pg.line(i, 0 + high/2, i, 128 - high/2)
            else:
                high = 0
            for j in range(128):
                if (j < high/2): ##256 - high/2.0): ## sky
                    pic.set(i, j, color(0, 0, 128))
                elif (j < 128 - high/2): ##256 - high and hcolor != None): ## hit
                    if (hcolor != None):
                        pic.set(i, j, hcolor)
                else: ## ground
                    pic.set(i, j, color(0, 128, 0))
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
        for wall in self.world.walls:
            ## if intersection, can't move
            v1 = wall.points[0]
            v2 = wall.points[1]
            v3 = wall.points[2]
            v4 = wall.points[3]
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
    
    def update(self):
        scale = self.world.scale
        tvx = self.vx * math.sin(-self.direction + math.pi/2) + self.vy * math.cos(-self.direction + math.pi/2)
        tvy = self.vx * math.cos(-self.direction + math.pi/2) - self.vy * math.sin(-self.direction + math.pi/2)
        ## proposed positions:
        px = self.x + tvx/250.0 * scale 
        py = self.y + tvy/250.0 * scale 
        pdirection = self.direction - self.va 
        ## check to see if collision
        ## bounding box:
        p1 = rotateAround(px, py, 30/250.0 * scale, pdirection + math.pi/4 + 0 * math.pi/2)
        p2 = rotateAround(px, py, 30/250.0 * scale, pdirection + math.pi/4 + 1 * math.pi/2)
        p3 = rotateAround(px, py, 30/250.0 * scale, pdirection + math.pi/4 + 2 * math.pi/2)
        p4 = rotateAround(px, py, 30/250.0 * scale, pdirection + math.pi/4 + 3 * math.pi/2)
        self.bounding_box[0] = p1
        self.bounding_box[1] = p2
        self.bounding_box[2] = p3
        self.bounding_box[3] = p4
        self.stalled = False
        for wall in self.world.walls:
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
        
        if (not self.stalled):
            ## if no intersection, make move
            self.x = px 
            self.y = py 
            self.direction = pdirection 
        else:
            self.direction += random.random(.2) - .1
        
        ## update sensors, camera:
        ## on right:
        p = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction + math.pi/8)
        hit = self.castRay(p[0], p[1], -self.direction + math.pi/2.0, self.max_ir * scale)
        if (hit != None):
            self.ir_sensors[0] = hit
        else:
            self.ir_sensors[0] = None
        
        p = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction - math.pi/8)
        hit = self.castRay(p[0], p[1], -self.direction + math.pi/2, self.max_ir * scale)
        if (hit != None):
            self.ir_sensors[1] = hit
        else:
            self.ir_sensors[1] = None
        
        ## camera:
        for i in range(256):
            angle = i/256.0 * 60 - 30    
            self.camera[i] = self.castRay(self.x, self.y, -self.direction + math.pi/2.0 - angle*math.pi/180.0, 1000)
         
    def draw(self, canvas):
        scale = self.world.scale
        sx = [0.05, 0.05, 0.07, 0.07, 0.09, 0.09, 0.07, 
              0.07, 0.05, 0.05, -0.05, -0.05, -0.07, 
              -0.08, -0.09, -0.09, -0.08, -0.07, -0.05, 
              -0.05]
        sy = [0.06, 0.08, 0.07, 0.06, 0.06, -0.06, -0.06, 
              -0.07, -0.08, -0.06, -0.06, -0.08, -0.07, 
              -0.06, -0.05, 0.05, 0.06, 0.07, 0.08, 0.06]
        if self.debug:
            ## bounding box:
            p1 = rotateAround(self.x, self.y, 30/250.0 * scale, self.direction + math.pi/4.0 + 0 * math.pi/2.0)
            p2 = rotateAround(self.x, self.y, 30/250.0 * scale, self.direction + math.pi/4.0 + 1 * math.pi/2.0)
            p3 = rotateAround(self.x, self.y, 30/250.0 * scale, self.direction + math.pi/4.0 + 2 * math.pi/2.0)
            p4 = rotateAround(self.x, self.y, 30/250.0 * scale, self.direction + math.pi/4.0 + 3 * math.pi/2.0)
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

        canvas.pushMatrix()
        canvas.translate(self.x, self.y)
        canvas.rotate(self.direction)
        ## body:
        if (self.stalled):
            canvas.fill(Color(128, 128, 128))
            canvas.stroke(Color(255, 255, 255))
        else:
            canvas.fill(self.robot_color)
            canvas.noStroke()
        
        points = []
        for i in range(len(sx)):
            points.append(Point(sx[i] * scale, sy[i] * scale))
        canvas.noStroke()
        polygon = Polygon(points)
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
        ## draw sensors
        ## right front IR
        ## position of start of sensor:
        p1 = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction + math.pi/8)
        ## angle of sensor:
        p2 = rotateAround(p1[0], p1[1], self.getIR(0) * self.max_ir * scale, self.direction)
        dist = self.distance(p1[0], p1[1], p2[0], p2[1])
        if (self.getIR(0) < 1.0):
            canvas.stroke(Color(255))
        else:
            canvas.noStroke()
        canvas.fill(Color(128, 0, 128, 64))
        arc = Arc((p1[0], p1[1]), dist, self.direction - .5, self.direction + .5)
        arc.draw(canvas)
        ## left front IR
        p1 = rotateAround(self.x, self.y, 25/250.0 * scale, self.direction - math.pi/8)
        p2 = rotateAround(p1[0], p1[1], self.getIR(1) * self.max_ir * scale, self.direction)
        dist = self.distance(p1[0], p1[1], p2[0], p2[1])
        if (self.getIR(1) < 1.0):
            canvas.stroke(Color(255))
        else:
            canvas.noStroke()
        canvas.fill(Color(128, 0, 128, 64))
        arc = Arc((p1[0], p1[1]), dist, self.direction - .5, self.direction + .5)
        arc.draw(canvas)

SIMULATION = None

def simulation(w, h, *robots):
    global SIMULATION
    SIMULATION = World(w, h)
    for robot in robots:
        SIMULATION.addRobot(robot)
    return SIMULATION

def forward(*args, **kwargs):
    SIMULATION.robots[0].forward(*args, **kwargs)

def backward(*args, **kwargs):
    SIMULATION.robots[0].backward(*args, **kwargs)

def turnLeft(*args, **kwargs):
    SIMULATION.robots[0].turnLeft(*args, **kwargs)

def turnRight(*args, **kwargs):
    SIMULATION.robots[0].turnRight(*args, **kwargs)

def stop(*args, **kwargs):
    SIMULATION.robots[0].stop(*args, **kwargs)

def sleep(seconds=1):
    SIMULATION.robots[0].sleep(seconds)
