import pdb
import numpy as np
import quaternion
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

from noise import snoise2 #more coherent than pnoise2


def generate_perlin_terrain(n,
        scale=25.0,
        octaves=4,
        persistence=0.2,
        lacunarity=2,
        seed=0):
    """
    n                  grid size, should be odd
    """
    assert n%2==1, "n should be odd"
    terrain = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = i / scale
            y = j / scale
            terrain[i][j] = snoise2(x,
                                    y,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024,
                                    repeaty=1024,
                                    base=seed)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()+1e-8)

    assert (terrain>=0).all()
    #plt.imshow(terrain)
    #plt.show()
   
    assert (terrain.flatten().reshape(n,n)==terrain).all()

    return terrain.flatten()

