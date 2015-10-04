from calysto.simulation import Bug, Simulation
import math

def makeSimulation():
    robot = Bug(550, 350, -math.pi/2)
    simulation = Simulation(600, 400, robot)
    return simulation
