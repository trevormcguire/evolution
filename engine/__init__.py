"""
Evolution Simulation Engine.

:Authors:
Trevor McGuire <trevormcguire13@gmail.com>
"""
import numpy as np

from sqlalchemy.orm import Session

from db.database import get_session
from engine.environment import GridWorld


# Organism -> Species -> Population
class Engine:
    """Simulation Engine.
    
    simulation start: x organisms of the common ancestor play a round.
    we cluster the weights into C clusters (kmeans). we split into C species.
    each cluster gets a random addition to its nn arch (arch search).

    best performing organism, find most similar N, those get to reproduce.
    
    """
    # holds the position of where each organism is in the world.
    # this is more efficient than storing each organism's position in the organism itself
    # because we are avoiding complex search operations.
    # this info could also just go straight into the engine itself.
    def __init__(self, world: GridWorld, populations: list, **kwargs):
        self.session: Session = next(get_session())
        self.world = world
        # self.world_map = WorldMap(world, populations)
        self.populations = populations  # this should be stored in the database...
        self.current_step = kwargs.get("current_step", 0)
        self.positions = kwargs.get("positions", {})  # {organism_id: (x, y)}

    def run_step(self):
        ...

    def run(self):
        ...
