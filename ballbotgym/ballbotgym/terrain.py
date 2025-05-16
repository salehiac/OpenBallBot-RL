import pdb
import numpy as np
import quaternion
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

from noise import snoise2 #more coherent than pnoise2


def generate_banded(
    n,
    alphas:list,#in degrees
    d=10,
    flatten=True,
    start_from=None,
    show=False):

    terrain_list=[]
    num_slopes=len(alphas)

    prev_x=n//2 if start_from is None else start_from
    prev_h=0

    band_sz=(n//2)//num_slopes if start_from is None else n//num_slopes

    print(band_sz)

    terrain_strip=np.zeros(n)

    for s_i in range(num_slopes):

        alpha_rad=alphas[s_i]*np.pi/180.0

        x1=prev_x+d
        h1=prev_h

        x2=start_from+(s_i+1)*band_sz-d
        r=(x2-x1)/np.cos(alpha_rad)
        h2=r*np.sin(alpha_rad)

        a=(h1-h2)/(x1-x2)
        b=h1-a*x1

        for xx in range(x1,x2+1):
            terrain_strip[xx]=a*xx+b
        terrain_strip[x2+1:x2+d]=h2

        prev_x=x2
        prev_h=h2

   
    terrain_strip[prev_x:]=prev_h

    terrain=terrain_strip.reshape(-1,1).repeat(n,axis=-1)


    if show:
        plt.plot(terrain_strip)
        plt.axis("equal")
        plt.grid("on")
        plt.show()
        #plt.imshow(terrain)
        #plt.show()

    return terrain.flatten() if flatten else terrain




def generate_perlin_terrain(n,
        scale=18.0,
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
    # Normalize to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()+1e-8)


    assert (terrain>=0).all()
    #plt.imshow(terrain)
    #plt.show()
   
    assert (terrain.flatten().reshape(n,n)==terrain).all()
    
    return terrain.flatten()

if __name__=="__main__":

    _max_angle=20
    _num_bands=5
    _alphas=(np.random.rand(_num_bands)-0.5)*2*_max_angle
    terr=generate_banded(n=273,alphas=_alphas,d=1,start_from=0,show=True)

