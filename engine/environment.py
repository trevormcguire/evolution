from dataclasses import dataclass

import numpy as np


def generate_blobs(h, w, num_clusters=5, cluster_std=3, norm: bool = True, precision: int = None):
    """Generate a 2D array with Gaussian blobs."""
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

    # Normalize
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
        cluster_std: int,
        max_qty: int = 1000,
        refresh_rate: float = 0.1,
    ):
        self.qty = generate_blobs(h, w, num_clusters=nc, cluster_std=cluster_std, norm=True)
        self.qty = np.round(self.qty * max_qty, 2)  # Scale to max quantity
        self.refresh_rate = refresh_rate

    def refresh(self):
        self.qty += (self.qty * self.refresh_rate)
        self.qty = np.clip(self.qty, a_min=0., a_max=None)

# generate blobs for each resource to determine its distribution across the world?
# if the value the magnitude? can multiple by some max value (1000) for the qty.
# # generate_blobs(30, 10, 5, 1, norm=True) seems to work well for 30x10 grid. 


class GridWorld:
    def __init__(self, h: int, w: int, num_resources: int = 10):
        self.h = h
        self.w = w
        self.grid = np.zeros((self.h, self.w))  # Initialize a grid with zeros
        self.resources = []
        for _ in range(num_resources):
            resource = ResourceMap(h=self.h, w=self.w, nc=np.random.randint(1, 5), cluster_std=np.random.uniform(0.5, 2.0))
            self.resources.append(resource)

    def step(self, population):
        """Simulate one step in the world."""
        for organism in population:
            # Update organism's position, collect resources, etc.
            pass

