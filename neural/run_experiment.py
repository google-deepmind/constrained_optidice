# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Experiment script for neural constrained OptiDICE."""

import functools
from typing import Any, Dict

from absl import app
from absl import flags
from acme.jax import networks as acme_networks
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import launchpad as lp

from constrained_optidice.neural.agent_distributed import DistributedCDICE
import constrained_optidice.neural.data_util as data_util
import constrained_optidice.neural.net_templates as net_templates
from constrained_optidice.neural.networks import CDICENetworks
import constrained_optidice.neural.rwrl as rwrl

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_evaluators', 1, 'Number of workers for evaluation.')
flags.DEFINE_string('data_path', None,
                    'Filepath for dataset used for training.')
flags.DEFINE_string('init_obs_data_path', None,
                    'Filepath for dataset used for training.')
flags.DEFINE_string('policy_extraction_mode', 'wsa',
                    'Policy extraction mode. (wsa, uniform).')
flags.DEFINE_integer('max_learner_steps', 100,
                     'The maximum number of training iteration.')
flags.DEFINE_float('gamma', 0.995,
                   'Discount factor. (0 ~ 1)')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size')
flags.DEFINE_float('alpha', 0.01, 'Regularizer on Df(d|dD).')
flags.DEFINE_float('cost_thresholds', 0.1, 'The cost constraint threshold.')
flags.DEFINE_string('f_type', 'softchi', 'The type of f-divergence function.')
flags.DEFINE_float('gradient_penalty', 1e-5,
                   'Gradient norm penalty regularization.')
flags.DEFINE_float('cost_ub_epsilon', 0.01,
                   'Adjusts the degree of overestimation of cost value.')
flags.DEFINE_string('task_name', 'rwrl:cartpole:realworld_swingup',
                    'Task name.')
flags.DEFINE_float('safety_coeff', 0.3,
                   'The safety coefficient for the RWRL task.')


def make_networks(observation_spec,
                  action_spec,
                  num_costs) -> CDICENetworks:
  """Create networks used by the agent."""
  make_networks_fn = functools.partial(
      net_templates.make_mix_gaussian_feedforward_networks,
      action_spec=action_spec,
      num_costs=num_costs)

  def _forward(obs):
    """Forward computation of (nu, lamb, chi, tau)."""
    networks = make_networks_fn()
    nu = networks['nu'](obs)[:, 0]
    lamb_params = hk.get_parameter('lamb_params', (num_costs,), init=jnp.zeros)
    lamb = jnp.clip(jnp.exp(lamb_params), 0, 1e3)

    chi = networks['chi'](obs)
    tau_params = hk.get_parameter('tau_params', (num_costs,), init=jnp.zeros)
    tau = jnp.exp(tau_params) + 1e-6

    return {
        'nu': nu,
        'lamb': lamb,
        'chi': chi,
        'tau': tau,
    }

  def _policy_fn(obs):
    """Policy returns action distribution."""
    networks = make_networks_fn()
    return networks['policy'](obs)

  def _behavior_fn(sample_key, obs):
    """Behavior returns action (will be used for evaluator)."""
    networks = make_networks_fn()
    return networks['behavior'](sample_key, obs)

  forward = hk.without_apply_rng(hk.transform(_forward))
  policy = hk.without_apply_rng(hk.transform(_policy_fn))
  behavior = hk.without_apply_rng(hk.transform(_behavior_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_obs = utils.add_batch_dim(utils.zeros_like(observation_spec))
  dummy_action = utils.add_batch_dim(utils.zeros_like(action_spec))
  dummy_sample_key = jax.random.PRNGKey(42)

  return CDICENetworks(
      forward=acme_networks.FeedForwardNetwork(
          init=lambda key: forward.init(key, dummy_obs),
          apply=forward.apply),
      policy=acme_networks.FeedForwardNetwork(
          init=lambda key: policy.init(key, dummy_obs), apply=policy.apply),
      behavior=acme_networks.FeedForwardNetwork(
          init=lambda key: behavior.init(key, dummy_sample_key, dummy_obs),
          apply=behavior.apply))


def get_program(num_evaluators: int, agent_params: Dict[str, Any],
                task_params: Dict[str, Any], seed: int) -> lp.Program:
  """Construct the program."""

  if task_params['task_name'].startswith('rwrl:'):
    _, domain_name, task_name = task_params['task_name'].split(':')
    environment_factory = functools.partial(
        rwrl.make_environment,
        domain_name=domain_name,
        task_name=task_name,
        safety_coeff=task_params['safety_coeff'])
    dataset_iterator_factory = functools.partial(
        data_util.create_data_iterators,
        data_path=task_params['data_path'],
        init_obs_data_path=task_params['init_obs_data_path'],
        batch_size=agent_params['batch_size']
    )
  else:
    raise NotImplementedError('Undefined task', task_params['task_name'])

  # Construct the program.
  program_builder = DistributedCDICE(
      environment_factory=environment_factory,
      dataset_iterator_factory=dataset_iterator_factory,
      task_name=task_params['task_name'],
      make_networks=make_networks,
      seed=seed,
      agent_params=agent_params,
      num_evaluators=num_evaluators)

  program = program_builder.build()

  return program


def main(unused_argv):
  # Get list of hyperparameter setting to run.
  agent_params = {
      'policy_extraction_mode': FLAGS.policy_extraction_mode,
      'max_learner_steps': FLAGS.max_learner_steps,
      'gamma': FLAGS.gamma,
      'learning_rate': FLAGS.learning_rate,
      'batch_size': FLAGS.batch_size,
      'alpha': FLAGS.alpha,
      'cost_thresholds': FLAGS.cost_thresholds,
      'f_type': FLAGS.f_type,
      'gradient_penalty': FLAGS.gradient_penalty,
      'cost_ub_epsilon': FLAGS.cost_ub_epsilon,
  }

  task_params = {
      'task_name': FLAGS.task_name,
      'safety_coeff': FLAGS.safety_coeff,
      'data_path': FLAGS.data_path,
      'init_obs_data_path': FLAGS.init_obs_data_path,
  }

  if FLAGS.data_path is None or FLAGS.init_obs_data_path is None:
    raise ValueError(
        'FLAGS.data_path and FLAGS.init_obs_data_path should be specified.')

  # Local launch for debugging.
  lp.launch(get_program(num_evaluators=FLAGS.num_evaluators,
                        agent_params=agent_params,
                        task_params=task_params,
                        seed=FLAGS.seed))


if __name__ == '__main__':
  app.run(main)
