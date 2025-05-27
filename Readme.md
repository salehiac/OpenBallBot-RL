# Reinforcement Learning for Ballbot control in uneven terrain

This repo contains the MuJoCo-based ballbot simulation as well as the RL code to reproduce the navigation results from the paper

```
Salehi, Achkan. "Reinforcement Learning for Ballbot Navigation in Uneven Terrain." arXiv preprint arXiv:2505.18417 (2025).
```

Here are some navigation examples from a trained policy on four different, *randomly sampled* uneven terrains:


<p float="left">
  <img src="/images/episode_1.gif" width="40.0%" />
  <img src="/images/episode_2.gif" width="40.0%" />
  <img src="/images/episode_3.gif" width="40.0%" />
  <img src="/images/episode_4.gif" width="29.0%" />
</p>

Note that in the above, eval episodes `1, 2, 3` are `4000` timesteps long, while episode `4` is `10000` steps long. The policy has been trained with a maximum of `4000` steps, therefore, this last evaluation can bee seen as a demonstration of generalization capability. 

## Warning!

Omniwheels are simulated using capsules with anisotropic friction. This requires a fix that is not (yet) part of the official MuJoCo release. Therefore, you **must** apply the provided patch
to your clone of MuJoCo and build both MuJoCo and the python bindings from source.

For more information, see [This discussion](https://github.com/google-deepmind/mujoco/discussions/2517).

## Build instructions

Make sure you have CMake and a C++17 compiler installed.

### Building MuJoCo from source (python bindings will be built separately)

    1. Clone the MuJoCo repository: `git clone https://github.com/deepmind/mujoco.git` and cd into it.
    2. This step is optional but **recommended** due to the patching issue mentioned above: `git checkout 99490163df46f65a0cabcf8efef61b3164faa620`
    3. Copy the patch `mujoco_fix.patch` provided in our repository to `<your_mujoco_repo>`, then `cd` into the latter and apply the patch: `patch -p1 < mujoco_fix.patch`

The rest of the instructions are identical to the [official MuJoCo guide](https://mujoco.readthedocs.io/en/latest/programming/#building-mujoco-from-source) for building from source:
    
    4. Create a new build directory and cd into it.
    5. Run `cmake $PATH_TO_CLONED_REPO` to configure the build.
    6. Run `cmake --build .` to build.
    7. Select the directory: `cmake $PATH_TO_CLONED_REPO -DCMAKE_INSTALL_PREFIX=<my_install_dir>`
    8. After building, install with `cmake --install . `

### Building the python bindings 

Once you have built the patched MuJoCo version from above, the steps for building the python bindings are almost identical to those from the [official MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/python.html#python-bindings):

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
pip install mujoco-x.y.z.tar.gz #replace x.y.z with the appropriate integers
```

NOTE: If you're using conda, you might need `conda install -c conda-forge libstdcxx` to avoid some gxx related issues.

### Other requirements

Make sure that you have a recent version of `pytorch` as well as a recent version of `stable_baselines3` installed. This code has been tested with torch version `'2.7.0+cu126'`.

Other requirements can be found in `requirements.txt`.


### Install the Ballbot Environment

```
cd OpenBallbot-RL/ballbotgym/
pip install -e .
```

### Sanity Check

To test that everything works well, run

```
cd OpenBallbot-RL/scripts
python3 test_pid.py
```

This uses a simple PID controller to balance the robot on flat terrain.

## Training an agent

Edit the `OpenBallbot-RL/config/train_ppo_directional.yaml` file if necessary, and then

```
cd scripts
python3  train.py --config ../config/train_ppo_directional.yaml
```

To see the progress of your training, you can use

```
python3 ../utils/plotting_tools.py --csv log/progress.csv --config log/config.yaml --plot_train
```

The default yaml config file should result in something that looks like 
<p float="left">
  <img src="/images/a.png" width="49.0%" />
  <img src="/images/b.png" width="49.0%" />
</p>

**Note**: The training process uses a pretrained depth-encoder, which is provided in `<root>/encoder_frozen/encoder_epoch_53`. If for some reason you prefer to train your own, you can use the `scripts/gather_data.py` and `sscripts/pretrain_encoder.py` scripts. 

## Evaluating an agent

You can see how the agent behaves using the `OpenBallbot-RL/scripts/test.py` script.

```
python3 test.py --algo ppo --n_test=<numer_of_tests_to_perform> --path <path_to_your_model>
```

## Trained policies

A trained policy is provided in the `OpenBallbot-RL/trained_agents/` directory, and can be tested using the line above.
