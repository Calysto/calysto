from calysto.simulation import LadyBug, Spider, Simulation
import math

def makeSimulation():
    ladybug = LadyBug(550, 350, -math.pi/2)
    spider = Spider(50, 50, 0)
    def spiderBrain():
        direction = 1
        while simulation.is_running.is_set():
            if spider.stalled:
                direction = direction * -1
            if direction == 1:
                spider.forward(1)
            else:
                spider.backward(1)
        spider.stop()
    spider.brain = spiderBrain
    simulation = Simulation(600, 400, ladybug, spider)
    simulation.makeWall(500, 100, 10, 200, "yellow")
    simulation.makeWall(10, 100, 190, 10, "yellow")
    simulation.makeWall(300, 100, 200, 10, "yellow")
    simulation.makeWall(100, 300, 410, 10, "yellow")
    simulation.makeWall(10, 200, 390, 10, "yellow")
    return simulation
