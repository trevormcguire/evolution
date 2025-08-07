from dataclasses import dataclass

import numpy as np


def generate_blobs(h, w, num_clusters=5, cluster_std=3, norm: bool = True, precision: int = None):
    """Generate a 2D array with Gaussian blobs."""
    # np.round(generate_blobs(30, 10, 5, 1, norm=True), 2)
    shape = (h, w)
    arr = np.zeros(shape)

    for _ in range(num_clusters):
        # randomly choose a center location 
        cx = np.random.randint(0, h)
        cy = np.random.randint(0, w)
        # Create grid of coordinates
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        # compute euclidean distance from center
        distance = np.sqrt((xv - cy)**2 + (yv - cx)**2)
        # 2D Gaussian function
        cluster = np.exp(-(distance**2) / (2 * cluster_std**2))
        arr += cluster

    # Normalize between 0 and 1
    if norm:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    if precision is not None and precision > 0:
        arr = np.round(arr, precision)
    return arr

class ResourceMap:
    def __init__(
        self,
        h: int,
        w: int,
        nc: int,
        std: int,
        initial_max_qty: int = 1000,
        refresh_rate: float = 0.1,
    ):
        # density is floats between 0 and 1
        self.density = generate_blobs(h, w, num_clusters=nc, cluster_std=std, norm=True, precision=2)
        self.qty = np.round(self.density * initial_max_qty, 2)  # Scale to max quantity
        self.refresh_rate = refresh_rate

    def refresh(self):
        self.qty += (self.qty * self.refresh_rate)
        self.qty = np.clip(self.qty, a_min=0., a_max=None)

# generate blobs for each resource to determine its distribution across the world?
# if the value the magnitude? can multiple by some max value (1000) for the qty.
# # generate_blobs(30, 10, 5, 1, norm=True) seems to work well for 30x10 grid. 

from enum import Enum
# its actions should lead to evolutionary paths, and specific adaptations (increased sight/horizon, speed, for example)
# if the organisms moved a bunch, reward can be better sight
# if the organism reproduced a bunch, a reward could be ... 
# better attributes (like sight) require more energy, which requires more resources
class ActionType(str, Enum):
    MOVE_LEFT = "move_left",
    MOVE_RIGHT = "move_right",
    MOVE_UP = "move_up",
    MOVE_DOWN = "move_down",
    NOTHING = "nothing"
    CONSUME = "consume"
    REPRODUCE = "reproduce"
# as organisms evolve, they may develop new actions or refine existing ones
# as organisms evolve, the number of resources they need to consume should increase

@dataclass
class ResourceConfig:
    n: int
    nc: int | tuple[int]
    std: int | tuple[float]

    def __post_init__(self):
        if isinstance(self.nc, tuple):
            self.nc = np.random.randint(self.nc[0], self.nc[1])
        if isinstance(self.std, tuple):
            self.std = np.random.uniform(self.std[0], self.std[1])

class GridWorld:
    def __init__(
        self,
        h: int,
        w: int,
        resource_config: ResourceConfig,
    ):
        self.h = h
        self.w = w
        self.resource_config = resource_config

        self.grid = np.zeros((self.h, self.w))  # Initialize a grid with zeros
        self.resources = [
            ResourceMap(self.h ,self.w, nc=self.resource_config.nc, std=self.resource_config.std) for _ in range(self.resource_config.n)
        ]


    def get_state_resource_map(self, x, y, horizon: int = 0, flatten: bool = True):
        """
        Return the resource quantities in a square region centered at (x, y) with the given horizon.
        For horizon=1, returns a 3x3 region; for horizon=2, a 5x5 region, etc.
        Output shape: (num_resources, num_positions)
        """
        region_size = 2 * horizon + 1
        offsets = []
        for dx in range(-horizon, horizon + 1):
            for dy in range(-horizon, horizon + 1):
                offsets.append((dx, dy))
        positions = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                positions.append((nx, ny))
            else:
                positions.append(None)  # Out of bounds

        states = []
        for r in self.resources:
            resource_vals = []
            for pos in positions:
                if pos is not None:
                    resource_vals.append(r.qty[pos[1], pos[0]])
                else:
                    resource_vals.append(0.0)  # Or np.nan if you prefer
            if not flatten:
                resource_vals = np.array(resource_vals).reshape(region_size, region_size)
            states.append(resource_vals)
        # Output: (num_resources, num_positions)
        return np.array(states)

    def get_states(self, x, y, horizon: int = 0, flatten: bool = True):
        # flatten to 1d for the neural nets for now?
        # always return horizon = 0? and if horizon = 2 always return 0 and 1, as well?
        # otherwise nn input layer won't be persisted. 
        # so we'll need to implement concat and backprop for concat. 
        state_resource_map = self.get_state_resource_map(x, y, horizon, flatten=flatten)
        return {"resource_map": state_resource_map}

    def organism_step(self, organism, action):
        ...

    def step(self, action: str):
        """Simulate one step in the world."""
        if action not in ActionType:
            raise ValueError(f"Invalid action: {action}")
        # # Update organism's position based on action
        # if action == ActionType.MOVE_LEFT:
        #     organism.position = (max(0, organism.position[0] - 1), organism.position[1])
        # elif action == ActionType.MOVE_RIGHT:


# class WorldMap:

    
#     def __init__(self, word: GridWorld, populations: list):
#         self.grid_world = word
#         self.populations = populations
