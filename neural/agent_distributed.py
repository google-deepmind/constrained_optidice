# Copyright 2021 DeepMind Technologies Limited
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

"""Offline constrained RL agent."""

import time
from typing import Any, Callable, Dict, Iterator, Tuple

from acme import core
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.jax import savers
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import dm_env
import jax
import launchpad as lp
import numpy as np
import reverb
import tree

from constrained_optidice.neural import cost
from constrained_optidice.neural import cost_environment_loop
from constrained_optidice.neural import learning
from constrained_optidice.neural.networks import CDICENetworks

NestedArraySpec = tree.Structure[dm_env.specs.Array]


class DistributedCDICE:
  """Program definition for COptiDICE."""

  def __init__(
      self,
      environment_factory: Callable[[], dm_env.Environment],
      dataset_iterator_factory: Callable[[],
                                         Tuple[Iterator[reverb.ReplaySample],
                                               Iterator[Dict[str,
                                                             np.ndarray]]]],
      task_name: str,
      make_networks: Callable[[NestedArraySpec, NestedArraySpec, int],
                              CDICENetworks],
      seed: int,
      agent_params: Dict[str, Any],
      num_evaluators: int = 1,
  ):
    # Define cost function per each task.
    self._cost_fn, self._num_costs = cost.domain_cost_fn(task_name)

    self._environment_factory = environment_factory
    self._dataset_iterator_factory = dataset_iterator_factory
    self._key = jax.random.PRNGKey(seed)
    self._learner_key, self._evaluator_key, self._video_recorder_key, self._key = jax.random.split(
        self._key, 4)
    self._task_name = task_name
    self._make_networks = make_networks
    self._num_evaluators = num_evaluators
    self._agent_params = agent_params

    environment = self._environment_factory()
    self._action_spec = environment.action_spec()
    self._obs_spec = environment.observation_spec()

  def counter(self):
    """The counter process."""
    return savers.CheckpointingRunner(
        counting.Counter(), subdirectory='counter', time_delta_minutes=5)

  def learner(self, counter: counting.Counter):
    """The learning process."""
    transition_dataset, init_obs_dataset = self._dataset_iterator_factory()

    # Make networks
    networks = self._make_networks(self._obs_spec, self._action_spec,
                                   self._num_costs)

    logger = loggers.make_default_logger('learner', time_delta=5.0)
    # Record steps with learner prefix.
    counter = counting.Counter(counter, prefix='learner')

    return learning.CDICELearner(
        key=self._learner_key,
        networks=networks,
        transition_dataset=transition_dataset,
        init_obs_dataset=init_obs_dataset,
        agent_params=self._agent_params,
        num_costs=self._num_costs,
        cost_fn=self._cost_fn,
        logger=logger,
        counter=counter)

  def evaluator(self,
                variable_source: core.VariableSource,
                counter: counting.Counter):
    """The evaluation process."""
    gamma = self._agent_params['gamma']
    environment = self._environment_factory()

    networks = self._make_networks(self._obs_spec, self._action_spec,
                                   self._num_costs)

    def actor_network(variables, key, obs):
      unused_params, policy_params = variables
      action = networks.behavior.apply(policy_params, key, obs)
      return action

    # Inference happens on CPU, so it's better to move variables there too.
    variable_client = variable_utils.VariableClient(
        variable_source,
        key=['params', 'policy_params'],
        device='cpu',
        update_period=1000)
    # Make sure not to evaluate random actor right after preemption.
    variable_client.update_and_wait()

    # Create the actor loading the weights from variable source.
    # (Actor network (params, key, obs) -> action)
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
        actor_network)
    actor = actors.GenericActor(
        actor_core, self._evaluator_key, variable_client, backend='cpu')

    # Wait until the learner starts to learn.
    while 'learner_steps' not in counter.get_counts():
      time.sleep(1)

    # Create the run loop and return it.
    logger = loggers.make_default_logger('evaluator', time_delta=5.0)
    return cost_environment_loop.CostEnvironmentLoop(environment, actor, gamma,
                                                     self._num_costs,
                                                     self._cost_fn, counter,
                                                     logger)

  def coordinator(self,
                  counter: counting.Counter,
                  max_steps: int,
                  steps_key: str = 'learner_steps'):
    return lp_utils.StepsLimiter(counter, max_steps, steps_key=steps_key)

  def build(self, name: str = 'cdice'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))
      if self._agent_params['max_learner_steps']:
        _ = program.add_node(
            lp.CourierNode(self.coordinator, counter,
                           self._agent_params['max_learner_steps'],
                           'learner_steps'))

    with program.group('learner'):
      learner = program.add_node(lp.CourierNode(self.learner, counter))

    with program.group('evaluator'):
      for _ in range(self._num_evaluators):
        program.add_node(lp.CourierNode(self.evaluator, learner, counter))

    return program
