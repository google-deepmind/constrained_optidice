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

"""A simple agent-environment interaction loop."""

import operator
import time
from typing import Optional, Callable

from acme import core
from acme.utils import counting
from acme.utils import loggers

import dm_env
from dm_env import specs
import numpy as np
import tree


class CostEnvironmentLoop(core.Worker):
  """A simple RL environment loop.

  This tracks cost return as well as reward return.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. Agent is updated if `should_update=True`. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      gamma: float,
      num_costs: int,
      cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'environment_loop',
  ):
    self._environment = environment
    self._actor = actor
    self._gamma = gamma
    self._num_costs = num_costs
    self._cost_fn = cost_fn
    if counter is None:
      counter = counting.Counter()
    if logger is None:
      logger = loggers.make_default_logger(label)
    self._counter = counter
    self._logger = logger
    self._should_update = should_update

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData` containing episode stats.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    cost_spec = specs.Array(shape=(self._num_costs,), dtype=float, name='cost')
    episode_cost_return = tree.map_structure(_generate_zeros_from_spec,
                                             cost_spec)
    gamma_sum = 0.0

    timestep = self._environment.reset()

    self._actor.observe_first(timestep)

    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      action = self._actor.select_action(timestep.observation)
      timestep = self._environment.step(action)

      # Compute a immediate cost for (obs, action).
      # cost_fn is defined in terms of batch input/output
      cost = self._cost_fn(
          tree.map_structure(lambda x: np.array([x]), timestep.observation),
          np.array([action]))[0]

      # Have the agent observe the timestep and let the actor update itself.
      self._actor.observe(action, next_timestep=timestep)
      if self._should_update:
        self._actor.update()

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd, episode_return,
                                          timestep.reward)

      episode_cost_return = tree.map_structure(operator.iadd,
                                               episode_cost_return, cost)
      gamma_sum += self._gamma**episode_steps
      episode_steps += 1

    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
    }

    for k, cost in enumerate(episode_cost_return):
      result[f'episode_cost_return_{k}'] = cost
      result[f'episode_average_cost_{k}'] = cost / episode_steps

    result.update(counts)
    return result

  def run(self,
          *,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If both
    num_episodes and num_steps arguments are provided, the first criterion met
    between the two will terminate the run loop.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return (num_episodes is not None and
              episode_count >= num_episodes) or (num_steps is not None and
                                                 step_count >= num_steps)

    episode_count = 0
    step_count = 0
    while not should_terminate(episode_count, step_count):
      result = self.run_episode()
      episode_count += 1
      step_count += result['episode_length']
      self._logger.write(result)


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
