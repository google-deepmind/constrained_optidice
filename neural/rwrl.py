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

"""RWRL environment builder."""

from typing import Optional

from acme import wrappers
from dm_control.rl import control
import dm_env
import numpy as np
import realworldrl_suite.environments as rwrl


# The hardest constraint for a specific task.
PER_DOMAIN_HARDEST_CONSTRAINT = {
    'cartpole': 'slider_pos_constraint',
    'humanoid': 'joint_angle_constraint',
    'quadruped': 'joint_angle_constraint',
    'walker': 'joint_velocity_constraint'}


def get_hardest_constraints_index(domain: str, env: control.Environment):
  try:
    constraint_name = PER_DOMAIN_HARDEST_CONSTRAINT[domain]
    return list(env._task.constraints).index(constraint_name)  # pylint: disable=protected-access
  except ValueError as err:
    raise ValueError('Invalid domain or domain unsupported') from err


class AddPredictionHeadsWrapper(wrappers.EnvironmentWrapper):

  @property
  def prediction_head_names(self):
    # The first prediction head should be 'reward'
    return ['reward', 'penalties']


class ConstraintsConverter(wrappers.EnvironmentWrapper):
  """Converts (bool) binary constraints to float penalties.

  This wrapper:
  - Extracts binary constraints from timestep.observation[from_key].
  - Flips them (negates them) if requested.
  - Keeps just a single constraint by index, if requested.
  - Converts them to floats, yielding penalties.
  - Stores the penalties in timestep.observation[to_key].
  """

  def __init__(self,
               environment: dm_env.Environment,
               from_key: str = 'constraints',
               flip: bool = True,
               keep_only_at_index: Optional[int] = None,
               to_key: str = 'penalties'):
    """Wrapper initializer.

    Args:
        environment (dm_env.Environment): Environment to wrap.
        from_key (str, optional): Name of constraint in timestep.observation
          which will be mapped into timestep.observation[to_key]
        flip (bool, optional): Whether to negate observation[from_key]
        keep_only_at_index (Optional[int], optional): Which individual
          constraint to select from observation[from_key]
        to_key (str, optional): Name of the key in timestep.observation where
          the updated constraint will be saved into.
    """
    super().__init__(environment)

    self._from_key = from_key
    self._keep_index = keep_only_at_index
    self._flip = flip
    self._to_key = to_key

  def step(self, action) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.step(action))

  def reset(self) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.reset())

  def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Pops, selects, flips, casts and stores updated constraints."""
    # Extract binary constraints
    binary_constraints = timestep.observation.pop(self._from_key, None)

    # Keep one constraint
    if self._keep_index is not None:
      i = self._keep_index
      binary_constraints = binary_constraints[i:i + 1]  # slice => 1-elem. array

    # Flip semantics (useful if they were is-satisfied style constraints)
    if self._flip:
      # I.e., to obtain: (0.0 == no penalty and 1.0 = penalty)
      binary_constraints = np.logical_not(binary_constraints)

    # Convert to penalties as floats
    timestep.observation[self._to_key] = binary_constraints.astype(np.float64)

    return timestep

  def observation_spec(self):
    """Alters the observation spec accordingly."""
    observation_spec = self._environment.observation_spec()

    # Convert binary constraints spec to a penalty spec
    # i.e. convert dtype from bool to float64
    constraints_spec = observation_spec.pop(self._from_key, None)
    updated_spec = constraints_spec.replace(dtype=np.float64)

    # Change spec to 1-element array if only one constraint is selected
    if self._keep_index is not None:
      updated_spec = updated_spec.replace(shape=(1,))

    observation_spec[self._to_key] = updated_spec
    return observation_spec


def make_environment(domain_name: str, task_name: str, safety_coeff: float):
  """Make RWRL environment with safety_spec."""
  safety_spec_dict = {
      'enable': True,
      'binary': True,
      'observations': True,
      'safety_coeff': safety_coeff
  }

  environment = rwrl.load(
      domain_name=domain_name,
      task_name=task_name,
      safety_spec=safety_spec_dict,
      environment_kwargs={'log_safety_vars': False, 'flat_observation': False})
  environment = ConstraintsConverter(
      environment,
      from_key='constraints',
      flip=True,
      keep_only_at_index=get_hardest_constraints_index(
          domain_name, environment),
      to_key='penalties')
  environment = AddPredictionHeadsWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment
