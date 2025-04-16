#OpenBallBot


Note that for anisotropic friction to work, we have to use the solution proposed by Yuval Tassa to my git issue: https://github.com/google-deepmind/mujoco/discussions/2517

Basically, 1) build mujoco from source 2) build the python bindings from source (while inside your conda env) 3) replace line 318 as he says. 

IMPORTANT: If I release this, I will need to make a git patch, unless they integrate it into their library
