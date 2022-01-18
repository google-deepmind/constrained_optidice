# Constrained Offline RL via stationary distribution correction estimation.

This repository contains an implementation of cost-conservative constrained
OptiDICE, from the paper "COptiDICE: Offline Constrained Reinforcement Learning
via Stationary Distribution Correction Estimation" by Jongmin Lee,
Cosmin Paduraru, Daniel J Mankowitz, Nicolas Heess, Doina Precup, Kee-Eung Kim,
and Arthur Guez.

## Installation

1. Ensure that `cmake` is installed. For example by running `apt-get install cmake`.
2. Install the [MuJoCo library](https://github.com/deepmind/mujoco), if not
already present. This is required for
the neural model experiment.
3. Install the Python dependencies with:

  ```shell
      pip install -r requirements.txt
  ```

  Alternatively, the convenience install script will do this step within a
  Python virtual env. Run this script once as follows:

  ```shell
  cd <parent directory of the git clone>
  constrained_optidice/install.sh
  ```

## How to run

Assuming the install script in step 3 above was used, running sample experiments
can be done with:

```shell
cd <parent directory of the git clone>
constrained_optidice/run.sh
```

The script executes the following commands within the virtual env:

### Tabular CMDP experiment
```shell
python3 -m constrained_optidice.tabular.run_random_cmdp_experiment
```


### Neural model experiment
```shell
python3 -m constrained_optidice.neural.run_experiment \
  --data_path="constrained_optidice/data_example/cartpole_0.3_example.npz" \
  --init_obs_data_path="constrained_optidice/data_example/cartpole_0.3_example.npz" \
  --safety_coeff=0.3 \
  --max_learner_steps=50 \
  --lp_launch_type=local_mp
```


## Disclaimer

This is not an official Google product.
