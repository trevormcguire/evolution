"""
Evolution Simulation Engine.

:Authors:
Trevor McGuire <trevormcguire13@gmail.com>
"""
import numpy as np
class Engine:
    ...


class WorldDynamics:
    def __init__(self, g: int = 9.81):
        self.g = g  # gravitational constant

class WorldMap:
    """
    Represents the world map in which the simulation takes place.
    It can be a grid, a graph, or any other structure that represents the environment.
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))  # Initialize a grid with zeros

        # init resources, obstacles, etc.
        self.resources = []
        self.obstacles = []

        # elevation can be float -1 to 1, where 0 is sea level

class GridWorld(object):
    """
    Represents the universe in which the simulation takes place.
    """
    def __init__(self, dynamics: WorldDynamics, width: int, height: int):
        self.dynamics = dynamics
        self.grid = np.zeros((width, height))
        self.entities = []

    def add_entity(self, entity):
        self.entities.append(entity)

    def simulate(self):
        for entity in self.entities:
            entity.update()


def build_neural_net():
    ...

class Organism:
    """Represents a single individual
    
    a population of compatible organisms is called a species.
    how to determine compatibility? cosine distance between weights? architectur must match.

    an organism is a neural network with a set of weights and biases.
    it has a fitness score that is determined by how well it performs in the environment.
    it can mutate its weights and biases to explore the environment.
    it can learn from its experiences to improve its performance via Reinforcement Learning.
    it can mate with other organisms to create offspring.
    """
    def __init__(self):
        self.exp = []  # experiences
        self.fitness = 0  # fitness score
        ...
    
    def mutate(self):
        """Randomly Permute the organism's weights and biases."""
        ... 

    def mate(self, other):
        """Combine the weights and biases of this organism with another to create an offspring."""
        ...

    def learn(self):
        ...


class Species:
    """Represents a group of organisms"""
    def __init__(self):
        ...


class Population:
    """Represents a collection of species"""
    def __init__(self):
        self.species = []

    def add_species(self, species: Species):
        self.species.append(species)

    def evolve(self):
        for species in self.species:
            # Implement evolution logic here
            pass

