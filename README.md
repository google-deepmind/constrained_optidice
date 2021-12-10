# Constrained Offline RL via stationary distribution correction estimation.

This repository contains an implementation of cost-conservative constrained
OptiDICE.

## Dependencies

See `requirements.txt`.

## How to run

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
