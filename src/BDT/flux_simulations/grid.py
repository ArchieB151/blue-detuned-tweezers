# -*- coding: utf-8 -*-
import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba.typed import List

__spec__ = [
    ('N', nb.int64),
    ('L', nb.float64),
    ('grid', nb.types.ListType(nb.types.ListType(nb.int64)))
]

@jitclass(__spec__)
class Grid:
    """
    A class representing a grid for collision detection and particle allocation.

    Attributes:
        N (int): The number of cells along each dimension of the grid.
        L (float): The length of the grid.
        grid (List[List[int]]): A 3D grid represented as a list of lists, where each element
                                contains the indices of particles allocated to that cell.

    Methods:
        __init__(self, _N, _L): Initializes a Grid object with the given number of cells and length.
        inBounds(self, x, y, z): Checks if the given coordinates are within the bounds of the grid.
        toIndex(self, x, y, z): Converts the given coordinates to the corresponding grid index.
        toXYZ(self, i): Converts the given grid index to the corresponding coordinates.
        allocateToGrid(self, pos): Allocates particles to the grid cells based on their positions.
        updateGrid(self, pos): Updates the grid by re-allocating particles based on their updated positions.
        computeCollision(self, pos, radius, cell1, cell2, velocity, masses): Computes collisions between particles
                                                                             in the given cells.
        computeCollisions(self, pos, radius, velocity, masses): Computes collisions between particles in the grid.
    """

    def __init__(self, _N, _L):
        self.N = _N
        self.L = _L
        self.grid = List([List([nb.types.int64(x) for x in range(0)])
                         for i in range(_N * _N * _N)])

    def inBounds(self, x, y, z):
        """
        Check if the given coordinates (x, y, z) are within the bounds of the grid.

        Parameters:
        - x (int): The x-coordinate.
        - y (int): The y-coordinate.
        - z (int): The z-coordinate.

        Returns:
        - bool: True if the coordinates are within the bounds, False otherwise.
        """
        return x >= 0 and x < self.N and y >= 0 and y < self.N and z >= 0 and z < self.N

    def toIndex(self, x, y, z):
        """
        Converts the given coordinates (x, y, z) to a linear index.

        Parameters:
        x (int): The x-coordinate.
        y (int): The y-coordinate.
        z (int): The z-coordinate.

        Returns:
        int: The linear index corresponding to the given coordinates.
        """
        return x + y * self.N + z * self.N * self.N

    def toXYZ(self, i):
        """
        Converts a linear index to XYZ coordinates in a 3D grid.

        Parameters:
        - i (int): The linear index of the grid cell.

        Returns:
        - tuple: A tuple containing the X, Y, and Z coordinates of the grid cell.
        """

        x = i % self.N
        y = (i // self.N) % self.N
        z = i // (self.N * self.N)
        return (x, y, z)

    def allocateToGrid(self, pos):
        """
        Allocates positions to the grid cells based on their coordinates.

        Parameters:
        - pos: numpy.ndarray
            The positions to be allocated to the grid cells.

        Returns:
        None
        """
        cellWidth = self.L / self.N
        celli = (pos[:, 0] // cellWidth + (pos[:, 1] // cellWidth) * self.N +
                 (pos[:, 2] // cellWidth) * self.N * self.N).astype(np.int64)
        for (index, i) in enumerate(celli):
            self.grid[i].append(index)

    def updateGrid(self, pos):
        """
        Update the grid based on the given positions.

        Parameters:
        - pos: numpy.ndarray
            The positions of the cells.

        Returns:
        None
        """
        cellWidth = self.L / self.N
        celli = (pos[:, 0] // cellWidth + (pos[:, 1] // cellWidth) * self.N +
                 (pos[:, 2] // cellWidth) * self.N * self.N).astype(np.int64)
        toUpdate = List()
        for (cellIndex, cell) in enumerate(self.grid):
            toRemove = List()
            for i in cell:
                if celli[i] != cellIndex:
                    toUpdate.append((i, celli[i]))
                    toRemove.append(i)
            for i in toRemove:
                self.grid[cellIndex].remove(i)

        for (i, celli) in toUpdate:
            self.grid[celli].append(i)

    @staticmethod
    def computeCollision(pos, radius, cell1, cell2, velocity, masses):
        """
        Computes the collision between particles in the given cells.

        Parameters:
        - pos (numpy.ndarray): Array of particle positions.
        - radius (numpy.ndarray): Array of particle radii.
        - cell1 (list): List of indices of particles in the first cell.
        - cell2 (list): List of indices of particles in the second cell.
        - velocity (numpy.ndarray): Array of particle velocities.
        - masses (numpy.ndarray): Array of particle masses.

        Returns:
        None
        """

        for (i, j) in zip(cell1, cell2):
            if i == j:
                continue
            relPos = pos[i, :] - pos[j, :]
            relDist2 = np.sum(relPos ** 2)
            if relDist2 > (radius[i] + radius[j]) ** 2:
                continue
            relVel = velocity[i, :] - velocity[j, :]
            vnorm = np.dot(relVel, relPos) / relDist2
            velScale = 2 * vnorm * relPos / (masses[i] + masses[j])
            velocity[i, :] -= masses[j] * velScale
            velocity[j, :] += masses[i] * velScale



    def computeCollisions(self, pos, radius, velocity, masses):
        """
        Computes collisions between particles in the grid.

        Args:
            pos (tuple): The position of the particle.
            radius (float): The radius of the particle.
            velocity (tuple): The velocity of the particle.
            masses (list): List of masses of particles in the grid.

        Returns:
            None
        """
        offsets = [(0, 1, 1), (1, 1, 1), (-1, 1, 1), (1, 0, 1), (-1, -1, 1), (0, -1, 1), (1, -1, 1),
                   (-1, 0, 1), (0, 0, 1), (0, 0, 0), (1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0)]
        for (index, cell1) in enumerate(self.grid):
            (i, j, k) = self.toXYZ(index)
            for offset in offsets:
                if not self.inBounds(i + offset[0], j + offset[1], k + offset[2]):
                    continue
                cell2 = self.grid[self.toIndex(
                    i + offset[0], j + offset[1], k + offset[2])]
                self.computeCollision(pos, radius, cell1, cell2, velocity, masses)
