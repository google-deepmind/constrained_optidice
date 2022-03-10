# Constrained Offline RL via stationary distribution correction estimation.

This repository contains an implementation of cost-conservative constrained
OptiDICE, from the paper: *COptiDICE: Offline Constrained Reinforcement Learning
via Stationary Distribution Correction Estimation* by **Jongmin Lee**, Cosmin
Paduraru, Daniel J Mankowitz, Nicolas Heess, Doina Precup, Kee-Eung Kim, and
Arthur Guez. Published as a conference paper at the International Conference on
Learning Representations (ICLR) 2022.

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

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the License. You may obtain a
copy of the Apache 2.0 license at: <https://www.apache.org/licenses/LICENSE-2.0>

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
<https://creativecommons.org/licenses/by/4.0/legalcode>

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
