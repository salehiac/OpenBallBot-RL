# Reinforcmenet Learning for Ballbot control in uneven terrain


<p float="left">
  <img src="/images/episode_1.gif" width="32.0%" />
  <img src="/images/episode_2.gif" width="32.0%" />
  <img src="/images/episode_3.gif" width="32.0%" />
</p>

This repo contains the mujoco-based ballbot simulation as well as the RL code to reproduce the navigation results from the paper

```
<add citation>
```

## Warning!

Omniwheels are simulated using capsules with anisotropic friction. This requires a fix that is not (yet) part of the official Mujoco release. Therefore, you **must** apply the provided patch
to your clone of mujoco and build both Mujoco and the python bindings from source.

For more information, see [This discussion](https://github.com/google-deepmind/mujoco/discussions/2517).

## Build instructions

Make sure you have CMake and a C++17 compiler installed.

### Building Mujoco from source (python bidings will be built separately)

    1. Clone the mujoco repository: `git clone https://github.com/deepmind/mujoco.git`
    2. This step is optional but **recommended** due to the patching issue mentioned above: `git checkout 99490163df46f65a0cabcf8efef61b3164faa620`
    3. Applyt the patch: `patch -p1 < mujoco_fix.patch`

The rest of the instructions are identitcal to the [official mujoco guide](https://mujoco.readthedocs.io/en/latest/programming/#building-mujoco-from-source) for building from source:
    
    4. Create a new build directory and cd into it.
    5. Run `cmake $PATH_TO_CLONED_REPO` to configure the build.
    6. Run `cmake --build .` to build.
    7. Select the directory: `cmake $PATH_TO_CLONED_REPO -DCMAKE_INSTALL_PREFIX=<my_install_dir>`
    8. After building, install with `cmake --install . `

### Building the python bindings 

Once you have built the patched Mujoco version from above, the steps for building the python buildings are almost identical to those from the [official Mujoco documentation](https://mujoco.readthedocs.io/en/stable/python.html#python-bindings):

1. Change to this directory:

```
cd <the_mujoco_repo_from_above>/mujoco/python
```

2. Create a virtual environment and activate it (I use conda, but whatever floats your boat)
3. Generate a source distribution tarball:

```
bash make_sdist.sh
```
This will generate many files in `<repo_clone_path>/mujoco/python/`, among which you'll find `mujoco-x.y.z.tar.gz`.
4. Run this:

```
cd dist
export MUJOCO_PATH=/PATH/TO/MUJOCO \
export MUJOCO_PLUGIN_PATH=/PATH/TO/MUJOCO_PLUGIN \
pip install mujoco-x.y.z.tar.gz
```

NOTE: If you're using conda, you'll need `conda install -c conda-forge libstdcxx` to avoid some gxx related issues.

## Training an agent

## Evaluating an agent


