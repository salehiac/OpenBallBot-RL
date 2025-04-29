#OpenBallBot


Note that for anisotropic friction to work, we have to use the solution proposed by Yuval Tassa to my git issue: https://github.com/google-deepmind/mujoco/discussions/2517

Basically, 1) build mujoco from source 2) build the python bindings from source (while inside your conda env) 3) replace line 318 as he says. 

IMPORTANT: If I release this, I will need to make a git patch, unless they integrate it into their library


#Building Mujoco from source (python bidings will be built separately)

//copy pasted from https://mujoco.readthedocs.io/en/latest/programming/#building-mujoco-from-source

To build MuJoCo from source, you will need CMake and a working C++17 compiler installed. The steps are:

    1. Clone the mujoco repository: `git clone https://github.com/deepmind/mujoco.git`
    2. Create a new build directory and cd into it.
    3. Run `cmake $PATH_TO_CLONED_REPO` to configure the build.
    4. Run `cmake --build .` to build.
    5. Select the directory: `cmake $PATH_TO_CLONED_REPO -DCMAKE_INSTALL_PREFIX=<my_install_dir>`
    6. After building, install with `cmake --install . `

#Building the python bindings 
//slighlty modified from https://mujoco.readthedocs.io/en/stable/python.html#python-bindings

Make sure you have CMake and a C++17 compiler installed.

1. Install mujoco from source (see above), no need for downloading prepackaged binaries, DMGs etc as they suggest 
2. cd into the mujoco repo you used above (to install from source) and ten

```
cd mujoco/python
```

3. Create a virtual environment and activate it (I use conda, they use venv in their example but whatever floats your boat)
4. Generate a source distribution tarball:

```
bash make_sdist.sh
```
This will generate many files in `<repo_clone_path>/mujoco/python/`, among which you'll find `mujoco-x.y.z.tar.gz`.

5. Run this:

#the export was missing in the docs
cd dist
export MUJOCO_PATH=/PATH/TO/MUJOCO \
export MUJOCO_PLUGIN_PATH=/PATH/TO/MUJOCO_PLUGIN \
pip install mujoco-x.y.z.tar.gz

6. NOTE: If you're using conda, you'll need `conda install -c conda-forge libstdcxx` to avoid some gxx related issues
