"""
================
simulation magic
================

"""


from metakernel import Magic, option
from calysto.display import display
from calysto.simulation import *

    
class SimulationMagic(Magic):

    def cell_simulation(self):
        """
        """
        if "def brain" in self.code:
            env = {}
            exec(self.code, env)
            brain = env["brain"]
            from calysto.display import display
            from calysto.simulation import DiscreteView, get_robot
            sim = DiscreteView("Ladybug1")
            robot = get_robot()

            def update(robot):
                ox, oy = robot.x, robot.y
                brain(robot)
                if (ox, oy) == (robot.x, robot.y):
                    robot.set_energy(robot.energy - 0.75) # cost of being alive
                    
            robot.update = lambda: update(robot)
            display(sim.sim_widgets)
            self.evaluate = False
        else:
            from calysto.display import display
            from calysto.simulation import DiscreteView, get_robot
            vsim = DiscreteView("Ladybug1")
            robot = get_robot()
            robot.rules = self.code
            display(vsim.sim_widgets)
            self.evaluate = False


def register_magics(kernel):
    kernel.register_magics(SimulationMagic)

def register_ipython_magics():
    from metakernel import IPythonKernel
    from IPython.core.magic import register_cell_magic
    kernel = IPythonKernel()
    magic = SimulationMagic(kernel)

    @register_cell_magic
    def simulation(line, cell):
        magic.code = cell
        magic.cell_simulation()
