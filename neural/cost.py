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

"""Cost function definitions."""
_DEFAULT_NUM_COSTS = 1


def _default_cost_fn(obs, unused_action):
  """Cost function C(s,a)."""
  return obs['penalties']


def domain_cost_fn(unused_domain_task_name):
  """Output cost function and the number of costs for the given task."""
  return _default_cost_fn, _DEFAULT_NUM_COSTS
