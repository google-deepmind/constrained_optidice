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

"""Data utility functions."""
import numpy as np
import reverb
import tree


def _unflatten(flat_data):
  """Converts a flat dict of numpy arrays to the batch tuple structure."""
  o_tm1 = {
      'penalties': flat_data['penalties_tm1'],
      'position': flat_data['position_tm1'],
      'velocity': flat_data['velocity_tm1'],
  }
  a_tm1 = flat_data['action_tm1']
  r_t = flat_data['reward_t']
  d_t = flat_data['discount_t']
  o_t = {
      'penalties': flat_data['penalties_t'],
      'position': flat_data['position_t'],
      'velocity': flat_data['velocity_t'],
  }
  return (o_tm1, a_tm1, r_t, d_t, o_t)


def _gen_batch_iterator(path, batch_size):
  with np.load(path) as flat_data:
    data = _unflatten(flat_data)
  unused_o_tm1, unused_a_tm1, r_t, unused_d_t, unused_o_t = data
  num_samples = len(r_t)

  while True:
    indices = np.random.randint(0, num_samples, (batch_size,))
    yield tree.map_structure(lambda x: x[indices], data)


def create_data_iterators(data_path, init_obs_data_path, batch_size):
  """Create data iterator used for training."""
  def gen_data_iterator():
    """Iterator for transition samples (o_tm1, a_tm1, r_t, d_t, o_t)."""
    for batch_data in _gen_batch_iterator(data_path, batch_size):
      batch_info = reverb.SampleInfo(
          key=0, probability=1., table_size=1, priority=1.)
      yield reverb.ReplaySample(info=batch_info, data=batch_data)

  def gen_initial_obs_iterator():
    """Iterator for initial observation samples."""
    for batch_data in _gen_batch_iterator(init_obs_data_path, batch_size):
      yield batch_data[0]  # 0: o_tm1

  return gen_data_iterator(), gen_initial_obs_iterator()
