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

"""CoptiDICE learner implementation."""

import functools
import time
from typing import Any, Dict, List, Optional, NamedTuple

from absl import logging
import acme
from acme.jax import networks as networks_lib
from acme.jax import utils as acme_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tree

from constrained_optidice.neural.networks import CDICENetworks

stop_gradient = jax.lax.stop_gradient


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  optimizer_state: optax.OptState  # for (nu, lamb, chi, tau) params
  policy_optimizer_state: optax.OptState
  params: networks_lib.Params  # (nu, lamb, chi, tau)
  target_params: networks_lib.Params  # target network of (params)
  policy_params: networks_lib.Params
  target_policy_params: networks_lib.Params
  key: networks_lib.PRNGKey
  steps: int


def conditional_update(new_tensors, old_tensors, is_time):
  """Checks whether to update the params and returns the correct params."""
  return jax.tree_multimap(lambda new, old: jax.lax.select(is_time, new, old),
                           new_tensors, old_tensors)


def periodic_update(new_tensors, old_tensors, steps, update_period: int):
  """Periodically switch all elements from a nested struct with new elements."""
  return conditional_update(
      new_tensors, old_tensors, is_time=steps % update_period == 0)


def get_f_divergence_fn(f_type: str):
  """Returns a function that computes the provided f-divergence type."""
  if f_type == 'chisquare':
    def f_fn(x):
      return 0.5 * (x - 1)**2

    def f_prime_inv_fn(x):
      return x + 1
  elif f_type == 'softchi':
    def f_fn(x):
      return jnp.where(x < 1,
                       x * (jnp.log(x + 1e-10) - 1) + 1, 0.5 * (x - 1)**2)

    def f_prime_inv_fn(x):
      return jnp.where(x < 0, jnp.exp(jnp.minimum(x, 0)), x + 1)
  elif f_type == 'kl':
    def f_fn(x):
      return x * jnp.log(x + 1e-10)

    def f_prime_inv_fn(x):
      return jnp.exp(x - 1)
  else:
    raise NotImplementedError('undefined f_fn', f_type)

  return f_fn, f_prime_inv_fn


class CDICELearner(acme.Learner):
  """CDICE learner."""

  _state: TrainingState

  def __init__(self,
               key: networks_lib.PRNGKey,
               networks: CDICENetworks,
               transition_dataset,
               init_obs_dataset,
               agent_params: Dict[str, Any],
               num_costs: int,
               cost_fn,
               target_update_period: int = 1000,
               clipping: bool = False,
               logger: Optional[loggers.Logger] = None,
               counter: Optional[counting.Counter] = None):
    self._agent_params = agent_params
    self._target_update_period = target_update_period
    self._cost_fn = cost_fn

    self._transition_iterator = transition_dataset
    if isinstance(transition_dataset, tf.data.Dataset):
      self._transition_iterator = iter(transition_dataset.as_numpy_iterator())

    self._init_obs_iterator = init_obs_dataset
    if isinstance(init_obs_dataset, tf.data.Dataset):
      self._init_obs_iterator = iter(init_obs_dataset.as_numpy_iterator())

    policy_extraction_mode = self._agent_params['policy_extraction_mode']
    learning_rate = self._agent_params['learning_rate']
    gamma = self._agent_params['gamma']
    alpha = self._agent_params['alpha']
    f_type = self._agent_params['f_type']
    gradient_penalty: float = self._agent_params['gradient_penalty']
    cost_ub_eps: float = self._agent_params['cost_ub_epsilon']

    c_hat = jnp.ones(num_costs) * self._agent_params['cost_thresholds']

    # The function definition for f-divergence.
    f_fn, f_prime_inv_fn = get_f_divergence_fn(f_type)

    optimizer = optax.adam(learning_rate)
    policy_optimizer = optax.adam(learning_rate)

    def _analytic_w(params, data):
      """Compute the closed-form solution of w."""
      o_tm1, unused_a_tm1, r_t, c_t, d_t, o_t = data
      f = networks.forward.apply(params, o_tm1)
      f_next = networks.forward.apply(params, o_t)
      e_nu_lamb = r_t - jnp.sum(c_t * stop_gradient(f['lamb']), axis=-1)
      e_nu_lamb += gamma * d_t * f_next['nu'] - f['nu']
      w_sa = jax.nn.relu(f_prime_inv_fn(e_nu_lamb / alpha))

      return f, f_next, e_nu_lamb, w_sa

    # Compute gradients with respect to the input
    @functools.partial(jax.vmap, in_axes=(None, 0))
    @functools.partial(jax.grad, argnums=1)
    def nu_grad_input(params, obs):
      """Forward computation of nu for a single sample: obs -> ()."""
      f = networks.forward.apply(params, acme_utils.add_batch_dim(obs))
      return f['nu'][0]

    @functools.partial(jax.vmap, in_axes=(None, 0))
    @functools.partial(jax.jacobian, argnums=1)
    def chi_grad_input(params, obs):
      """Forward computation of nu for a single sample: obs -> ()."""
      f = networks.forward.apply(params, acme_utils.add_batch_dim(obs))
      return f['chi'][0]

    def _compute_obs_mix(obs1, obs2, eps):
      """Compute eps * obs1 + (1 - eps) * obs2."""
      e = tree.map_structure(lambda x, eps=eps: eps, obs1)
      return tree.map_structure(lambda x0, x1, e: (x0.T * e + x1.T * (1 - e)).T,
                                obs1, obs2, e)

    def loss(params: networks_lib.Params, data, init_o, key) -> jnp.ndarray:
      # Compute losses
      o_tm1, a_tm1, unused_r_t, c_t, d_t, unused_o_t = data
      f_init = networks.forward.apply(params, init_o)
      f, f_next, e_nu_lamb, w_sa = _analytic_w(params, data)
      w_sa_no_grad = stop_gradient(w_sa)

      # Gradient norm for o_mix: interpolate init_o and o_tm1 with eps~U(0,1)
      eps = jax.random.uniform(key, shape=(a_tm1.shape[0],))
      obs_mix = _compute_obs_mix(init_o, o_tm1, eps)

      nu_grad_norm = jnp.linalg.norm(  # 1e-10 was added to prevent nan
          acme_utils.batch_concat(nu_grad_input(params, obs_mix)) + 1e-10,
          axis=1)  # [batch_size]
      chi_grad_norm = jnp.linalg.norm(
          acme_utils.batch_concat(  # 1e-10 was added to prevent nan
              chi_grad_input(params, obs_mix), num_batch_dims=2) + 1e-10,
          axis=2)  # [batch_size, num_costs]

      # (chi, tau) loss
      batch_size = a_tm1.shape[0]
      if cost_ub_eps == 0:
        ell = jnp.zeros((batch_size, num_costs))
        chi_tau_loss = kl_divergence = 0
        cost_ub = jnp.mean(w_sa[:, None] * c_t, axis=0)
      else:
        ell = (1 - gamma) * f_init['chi']  # [n, num_costs]
        ell += w_sa_no_grad[:, None] * (
            c_t + gamma * d_t[:, None] * f_next['chi'] - f['chi'])
        logits = ell / stop_gradient(f['tau'])
        weights = jax.nn.softmax(logits, axis=0) * batch_size  # [n, num_costs]
        log_weights = jax.nn.log_softmax(logits, axis=0) + jnp.log(batch_size)
        kl_divergence = jnp.mean(
            weights * log_weights - weights + 1, axis=0)  # [num_costs]
        cost_ub = jnp.mean(weights * w_sa_no_grad[:, None] * c_t, axis=0)
        chi_tau_loss = jnp.sum(jnp.mean(weights * ell, axis=0))
        chi_tau_loss += jnp.sum(-f['tau'] *
                                (stop_gradient(kl_divergence) - cost_ub_eps))
        chi_tau_loss += gradient_penalty * jnp.mean(
            jnp.sum(jax.nn.relu(chi_grad_norm - 5)**2, axis=1), axis=0)  # GP

      # nu loss
      nu_loss = (1 - gamma) * jnp.mean(f_init['nu'])
      nu_loss += -alpha * jnp.mean(f_fn(w_sa))
      nu_loss += jnp.mean(w_sa * e_nu_lamb)
      nu_loss += gradient_penalty * jnp.mean(jax.nn.relu(nu_grad_norm - 5)**2)

      # lamb loss
      lamb_loss = -jnp.dot(f['lamb'], stop_gradient(cost_ub) - c_hat)

      total_loss = nu_loss + lamb_loss + chi_tau_loss

      metrics = {
          'nu_loss': nu_loss,
          'lamb_loss': lamb_loss,
          'chi_tau_loss': chi_tau_loss,
          'nu': jnp.mean(f['nu']),
          'next_nu': jnp.mean(f_next['nu']),
          'initial_nu': jnp.mean(f_init['nu']),
          'w_sa': jnp.mean(w_sa),
          'cost_ub': cost_ub,
          'kl_divergence': kl_divergence,
          'chi': jnp.mean(f['chi'], axis=0),
          'tau': f['tau'],
          'lamb': f['lamb'],
      }
      return total_loss, metrics

    def policy_loss(policy_params: networks_lib.Params,
                    params: networks_lib.Params,
                    data) -> jnp.ndarray:
      o_tm1, a_tm1, unused_r_t, unused_c_t, unused_d_t, unused_o_t = data
      pi_a_tm1 = networks.policy.apply(policy_params, o_tm1)

      # weighted BC
      assert len(pi_a_tm1.batch_shape) == 1
      logp_tm1 = pi_a_tm1.log_prob(a_tm1)
      if policy_extraction_mode == 'uniform':
        policy_loss = -jnp.mean(logp_tm1)  # vanilla BC
      elif policy_extraction_mode == 'wsa':
        _, _, _, w_sa = _analytic_w(params, data)
        assert len(w_sa.shape) == 1
        policy_loss = -jnp.mean(w_sa * logp_tm1)
      else:
        raise NotImplementedError('undefined policy extraction.',
                                  policy_extraction_mode)
      metrics = {'policy_loss': policy_loss}
      return policy_loss, metrics

    loss_grad = jax.grad(loss, has_aux=True)
    policy_loss_grad = jax.grad(policy_loss, has_aux=True)

    def _step(state: TrainingState, data, init_o):
      metrics = {}

      # Compute loss and gradients
      key, key_input = jax.random.split(state.key)
      loss_grads, info = loss_grad(state.params, data, init_o, key_input)
      policy_loss_grads, policy_info = policy_loss_grad(state.policy_params,
                                                        state.params,
                                                        data)
      metrics.update(info)
      metrics.update(policy_info)

      # Apply gradients
      updates, optimizer_state = optimizer.update(loss_grads,
                                                  state.optimizer_state)
      params = optax.apply_updates(state.params, updates)
      policy_updates, policy_optimizer_state = policy_optimizer.update(
          policy_loss_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, policy_updates)

      # Update training state
      target_params = periodic_update(params, state.target_params, state.steps,
                                      self._target_update_period)
      target_policy_params = periodic_update(policy_params,
                                             state.target_policy_params,
                                             state.steps,
                                             self._target_update_period)
      new_state = TrainingState(
          optimizer_state=optimizer_state,
          policy_optimizer_state=policy_optimizer_state,
          params=params,
          target_params=target_params,
          policy_params=policy_params,
          target_policy_params=target_policy_params,
          key=key,
          steps=state.steps + 1)

      return new_state, metrics

    @jax.jit
    def make_initial_state(key: networks_lib.PRNGKey) -> TrainingState:
      """Initializes the training state (parameters and optimizer state)."""
      key_nu, key_policy, key = jax.random.split(key, 3)
      # Initialize parameters
      params = networks.forward.init(key_nu)
      policy_params = networks.policy.init(key_policy)
      # Initialize optimizer states
      optimizer_state = optimizer.init(params)
      policy_optimizer_state = policy_optimizer.init(policy_params)
      # Define a training state
      state = TrainingState(
          optimizer_state=optimizer_state,
          policy_optimizer_state=policy_optimizer_state,
          params=params,
          target_params=params,
          policy_params=policy_params,
          target_policy_params=policy_params,
          key=key,
          steps=0)
      return state

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter(prefix='learner')
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Create initial state.
    self._state = make_initial_state(key)
    self._step = jax.jit(_step)

    self._timestamp = None

  def step(self):
    """Take one SGD step in the learner."""
    init_o = next(self._init_obs_iterator)

    sample = next(self._transition_iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = sample.data
    c_t = self._cost_fn(o_tm1, a_tm1)
    data = (o_tm1, a_tm1, r_t, c_t, d_t, o_t)

    # Gradient update
    new_state, metrics = self._step(self._state, data, init_o)
    self._state = new_state

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    self._logger.write({
        **metrics,
        **counts
    })

  def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
    variables = {
        'params': self._state.target_params,
        'policy_params': self._state.target_policy_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    logging.info('learner.save is called.')
    return self._state

  def restore(self, state: TrainingState):
    logging.info('learner.restore is called.')
    self._state = state
