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

"""Network architectures."""

from typing import Callable, Optional

from acme import specs
from acme.jax import networks as acme_networks
from acme.jax import utils as acme_utils
import haiku as hk
import jax.numpy as jnp
import numpy as np

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

uniform_initializer = hk.initializers.VarianceScaling(
    mode='fan_out', scale=1. / 3.)


class ResidualLayerNormWrapper(hk.Module):
  """Wrapper that applies residual connections and layer norm."""

  def __init__(self, layer: Callable[[jnp.ndarray], jnp.ndarray]):
    """Creates the Wrapper Class.

    Args:
      layer: module to wrap.
    """

    super().__init__(name='ResidualLayerNormWrapper')
    self._layer = layer

    self._layer_norm = hk.LayerNorm(
        axis=-1, create_scale=True, create_offset=True)

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Returns the result of the residual and layernorm computation.

    Args:
      inputs: inputs to the main module.
    """

    # Apply main module.
    outputs = self._layer(inputs)
    outputs = self._layer_norm(outputs + inputs)

    return outputs


class LayerNormAndResidualMLP(hk.Module):
  """MLP with residual connections and layer norm."""

  def __init__(self, hidden_size: int, num_blocks: int):
    """Create the model.

    Args:
      hidden_size: width of each hidden layer.
      num_blocks: number of blocks, each block being MLP([hidden_size,
        hidden_size]) + layer norm + residual connection.
    """
    super().__init__(name='LayerNormAndResidualMLP')

    # Create initial MLP layer.
    layers = [hk.nets.MLP([hidden_size], w_init=uniform_initializer)]

    # Follow it up with num_blocks MLPs with layernorm and residual connections.
    for _ in range(num_blocks):
      mlp = hk.nets.MLP([hidden_size, hidden_size], w_init=uniform_initializer)
      layers.append(ResidualLayerNormWrapper(mlp))

    self._network = hk.Sequential(layers)

  def __call__(self, inputs: jnp.ndarray):
    return self._network(inputs)


class UnivariateGaussianMixture(acme_networks.GaussianMixture):
  """Head which outputs a Mixture of Gaussians Distribution."""

  def __init__(self,
               num_dimensions: int,
               num_components: int = 5,
               init_scale: Optional[float] = None):
    """Create an mixture of Gaussian actor head.

    Args:
      num_dimensions: dimensionality of the output distribution. Each dimension
        is going to be an independent 1d GMM model.
      num_components: number of mixture components.
      init_scale: the initial scale for the Gaussian mixture components.
    """
    super().__init__(num_dimensions=num_dimensions,
                     num_components=num_components,
                     multivariate=False,
                     init_scale=init_scale,
                     name='UnivariateGaussianMixture')


class StochasticSamplingHead(hk.Module):
  """Simple haiku module to sample from a tfd.Distribution."""

  def __call__(self, sample_key: acme_networks.PRNGKey,
               distribution: tfd.Distribution):
    return distribution.sample(seed=sample_key)


def make_mix_gaussian_feedforward_networks(action_spec: specs.BoundedArray,
                                           num_costs: int):
  """Makes feedforward networks with mix gaussian actor head."""
  action_dim = np.prod(action_spec.shape, dtype=int)

  hidden_size = 1024

  nu_network = hk.Sequential([
      acme_utils.batch_concat,
      acme_networks.LayerNormMLP(layer_sizes=[512, 512, 256, 1]),
  ])

  chi_network = hk.Sequential([
      acme_utils.batch_concat,
      acme_networks.LayerNormMLP(layer_sizes=[512, 512, 256, num_costs]),
  ])

  actor_encoder = hk.Sequential([
      acme_utils.batch_concat,
      hk.Linear(300, w_init=uniform_initializer),
      hk.LayerNorm(slice(1, None), True, True),
      jnp.tanh,
  ])
  actor_neck = LayerNormAndResidualMLP(hidden_size, num_blocks=4)
  actor_head = UnivariateGaussianMixture(
      num_components=5, num_dimensions=action_dim)

  stochastic_policy_network = hk.Sequential(
      [actor_encoder, actor_neck, actor_head])

  class LowNoisePolicyNetwork(hk.Module):

    def __call__(self, inputs):
      x = actor_encoder(inputs)
      x = actor_neck(x)
      x = actor_head(x, low_noise_policy=True)
      return x

  low_noise_policy_network = LowNoisePolicyNetwork()

  # Behavior networks output an action while the policy outputs a distribution.
  stochastic_sampling_head = StochasticSamplingHead()

  class BehaviorNetwork(hk.Module):

    def __call__(self, sample_key, inputs):
      dist = low_noise_policy_network(inputs)
      return stochastic_sampling_head(sample_key, dist)

  behavior_network = BehaviorNetwork()

  return {
      'nu': nu_network,
      'chi': chi_network,
      'policy': stochastic_policy_network,
      'low_noise_policy': low_noise_policy_network,
      'behavior': behavior_network,
  }
