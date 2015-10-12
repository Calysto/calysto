from calysto.simulation import *
import random
import math

class LadybugSimulation(DiscreteSimulation):
    def initialize(self):
        super(LadybugSimulation, self).initialize()
        self.reset()
        for robot in self.robots:
            robot.state = "0"
        self.items["w"] = self.drawWall
        for i in range(self.psize[0]):
            self.setPatch(i, 0, "w")
            self.setPatch(i, self.psize[1] - 1, "w")
        for i in range(self.psize[1]):
            self.setPatch(0, i, "w")
            self.setPatch(self.psize[0] - 1, i, "w")
        self.addCluster(random.random() * 15, 
                        random.random() * 10, 
                        'f', 20)
        self.addCluster(random.random() * 15 + 30, 
                        random.random() * 10 + 20, 
                        'f', 20)

def makeSimulation():
    sim = LadybugSimulation(600, 400, 
                            draw_walls=False, 
                            background_color="green")
    ladybug = DiscreteLadybug(30, 20, -math.pi/2)
    sim.addRobot(ladybug)
    return sim
