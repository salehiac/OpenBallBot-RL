import pdb
import numpy as np
import quaternion
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

from noise import pnoise2


def generate_perlin_terrain(n,
                            flat_center_size=0,
                            scale=12.0,
                            octaves=6,
                            persistence=0.5,
                            lacunarity=2.0,
                            seed=0):
    """
    n                  grid size, should be odd
    flat_center_size   if non-zeros, a flat_center_size**2 are in the center will be set to zero height
    """
    assert n%2==1, "n should be odd"
    terrain = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = i / scale
            y = j / scale
            terrain[i][j] = pnoise2(x,
                                    y,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024,
                                    repeaty=1024,
                                    base=seed)
    # Normalize to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()+1e-8)

    if flat_center_size:
        cc=terrain.shape[0]//2
        dd=flat_center_size//2
        terrain[cc-dd:cc+dd,
                cc-dd:cc+dd]=np.minimum(terrain[cc-dd:cc+dd, cc-dd:cc+dd], 0.5)

    assert (terrain>=0).all()
    #plt.imshow(terrain)
    #plt.show()
    
    return terrain.flatten()
