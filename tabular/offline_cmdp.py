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

"""Implementation of tabular offline (C)MDP methods."""
import copy
import time

from absl import logging
import cvxopt
import jax
import jax.config
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.optimize
from constrained_optidice.tabular import mdp_util as util


cvxopt.solvers.options['show_progress'] = False
jax.config.update('jax_enable_x64', True)


def _compute_marginal_distribution(mdp, pi, regularizer=0):
  """Compute marginal distribution for the given policy pi, d^pi(s,a)."""
  p0_s = np.zeros(mdp.num_states)
  p0_s[mdp.initial_state] = 1
  p0 = (p0_s[:, None] * pi).reshape(mdp.num_states * mdp.num_actions)
  p_pi = (mdp.transition.reshape(mdp.num_states * mdp.num_actions,
                                 mdp.num_states)[:, :, None] * pi).reshape(
                                     mdp.num_states * mdp.num_actions,
                                     mdp.num_states * mdp.num_actions)
  d = np.ones(mdp.num_states * mdp.num_actions)
  d /= np.sum(d)
  d_diag = np.diag(d)
  e = np.sqrt(d_diag) @ (
      np.eye(mdp.num_states * mdp.num_actions) - mdp.gamma * p_pi)

  q = np.linalg.solve(
      e.T @ e + regularizer * np.eye(mdp.num_states * mdp.num_actions),
      (1 - mdp.gamma) * p0)
  w = q - mdp.gamma * p_pi @ q

  assert np.all(w > -1e-6), w
  d_pi = w * d
  d_pi[w < 0] = 0
  d_pi /= np.sum(d_pi)
  return d_pi.reshape(mdp.num_states, mdp.num_actions)


def generate_baseline_policy(cmdp: util.CMDP,
                             behavior_cost_thresholds: np.ndarray,
                             optimality: float) -> np.ndarray:
  """Generate a baseline policy for the CMDP.

  Args:
    cmdp: a CMDP instance.
    behavior_cost_thresholds: cost threshold for behavior policy. [num_costs]
    optimality: optimality of behavior policy.
      (0: uniform policy, 1: optimal policy)

  Returns:
    behavior policy. [num_states, num_actions]
  """
  cmdp = copy.copy(cmdp)
  cmdp.cost_thresholds = behavior_cost_thresholds
  cmdp_no_reward = copy.copy(cmdp)
  cmdp_no_reward.reward *= 0

  pi_opt = util.solve_cmdp(cmdp)
  pi_unif = np.ones((cmdp.num_states, cmdp.num_actions)) / cmdp.num_actions
  v_opt = util.policy_evaluation(cmdp, pi_opt)[0][0]
  q_opt = util.policy_evaluation(cmdp, pi_opt)[1]
  v_unif = util.policy_evaluation(cmdp, pi_unif)[0][0]

  v_final_target = v_opt * optimality + (1 - optimality) * v_unif

  softmax_reduction_factor = 0.9
  temperature = 1e-6
  pi_soft = scipy.special.softmax(q_opt / temperature, axis=1)
  while util.policy_evaluation(cmdp, pi_soft)[0][0] > v_final_target:
    temperature /= softmax_reduction_factor
    pi_soft = scipy.special.softmax(q_opt / temperature, axis=1)
    pi_soft /= np.sum(pi_soft, axis=1, keepdims=True)
    pi_soft = constrained_optidice(cmdp_no_reward, pi_soft, alpha=1)
    r, _, c, _ = util.policy_evaluation(cmdp, pi_soft)
    logging.info('temp=%.6f, R=%.3f, C=%.3f / v_opt=%.3f, f_target=%.3f',
                 temperature, r[0], c[0][0], v_opt, v_final_target)

  assert np.all(pi_soft >= -1e-4)

  pi_b = pi_soft.copy()
  return pi_b


def optidice(mdp: util.MDP, pi_b: np.ndarray, alpha: float):
  """f-divergence regularized RL.

  max_{d} E_d[R(s,a)] - alpha * E_{d_b}[f(d(s,a)/d_b(s,a))]

  We assume that f(x) = 0.5 (x-1)^2.

  Args:
    mdp: a MDP instance.
    pi_b: behavior policy. [num_states, num_actions]
    alpha: regularization hyperparameter for f-divergence.

  Returns:
    the resulting policy. [num_states, num_actions]
  """
  d_b = _compute_marginal_distribution(mdp, pi_b).reshape(
      mdp.num_states * mdp.num_actions) + 1e-6  # |S||A|
  d_b /= np.sum(d_b)
  p0 = np.eye(mdp.num_states)[mdp.initial_state]  # |S|
  r = np.array(mdp.reward.reshape(mdp.num_states * mdp.num_actions))
  p = np.array(
      mdp.transition.reshape(mdp.num_states * mdp.num_actions, mdp.num_states))
  p = p / np.sum(p, axis=1, keepdims=True)
  b = np.repeat(np.eye(mdp.num_states), mdp.num_actions, axis=0)  # |S||A| x |S|

  # Solve:
  # minimize    (1/2)*x^T P x + q^T x
  # subject to  G x <= h
  #             A x = b.
  d_diag = np.diag(d_b)
  qp_p = alpha * (d_diag)
  qp_q = -d_diag @ r - alpha * d_b
  qp_g = -np.eye(mdp.num_states * mdp.num_actions)
  qp_h = np.zeros(mdp.num_states * mdp.num_actions)
  qp_a = (b.T - mdp.gamma * p.T) @ d_diag
  qp_b = (1 - mdp.gamma) * p0
  cvxopt.solvers.options['show_progress'] = False
  res = cvxopt.solvers.qp(
      cvxopt.matrix(qp_p), cvxopt.matrix(qp_q), cvxopt.matrix(qp_g),
      cvxopt.matrix(qp_h), cvxopt.matrix(qp_a), cvxopt.matrix(qp_b))
  w = np.array(res['x'])[:, 0]  # [num_states * num_actions]
  assert np.all(w >= -1e-4), w
  w = np.clip(w, 1e-10, np.inf)
  pi = (w * d_b).reshape(mdp.num_states, mdp.num_actions) + 1e-10
  pi /= np.sum(pi, axis=1, keepdims=True)

  return w, d_b, pi


def constrained_optidice(cmdp: util.CMDP,
                         pi_b: np.ndarray,
                         alpha: float):
  """f-divergence regularized constrained RL.

  max_{d} E_d[R(s,a)] - alpha * E_{d_b}[f(d(s,a)/d_b(s,a))]
  s.t. E_d[C(s,a)] <= hat{c}.

  We assume that f(x) = 0.5 (x-1)^2.

  Args:
    cmdp: a CMDP instance.
    pi_b: behavior policy.
    alpha: regularization hyperparameter for f-divergence.

  Returns:
    the resulting policy. [num_states, num_actions]
  """
  d_b = _compute_marginal_distribution(cmdp, pi_b).reshape(
      cmdp.num_states * cmdp.num_actions) + 1e-6  # |S||A|
  d_b /= np.sum(d_b)
  p0 = np.eye(cmdp.num_states)[cmdp.initial_state]  # |S|
  p = np.array(
      cmdp.transition.reshape(cmdp.num_states * cmdp.num_actions,
                              cmdp.num_states))
  p = p / np.sum(p, axis=1, keepdims=True)
  b = np.repeat(
      np.eye(cmdp.num_states), cmdp.num_actions, axis=0)  # |S||A| x |S|
  r = np.array(cmdp.reward.reshape(cmdp.num_states * cmdp.num_actions))
  c = np.array(
      cmdp.costs.reshape(cmdp.num_costs, cmdp.num_states * cmdp.num_actions))

  # Solve:
  # minimize    (1/2)*x^T P x + q^T x
  # subject to  G x <= h
  #             A x = b.
  d_diag = np.diag(d_b)
  qp_p = alpha * (d_diag)
  qp_q = -d_diag @ r - alpha * d_b
  qp_g = np.concatenate(
      [c @ d_diag, -np.eye(cmdp.num_states * cmdp.num_actions)], axis=0)
  qp_h = np.concatenate(
      [cmdp.cost_thresholds,
       np.zeros(cmdp.num_states * cmdp.num_actions)])
  qp_a = (b.T - cmdp.gamma * p.T) @ d_diag
  qp_b = (1 - cmdp.gamma) * p0
  res = cvxopt.solvers.qp(
      cvxopt.matrix(qp_p), cvxopt.matrix(qp_q), cvxopt.matrix(qp_g),
      cvxopt.matrix(qp_h), cvxopt.matrix(qp_a), cvxopt.matrix(qp_b))
  w = np.array(res['x'])[:, 0]  # [num_states * num_actions]
  assert np.all(w >= -1e-4), w
  w = np.clip(w, 1e-10, np.inf)
  pi = (w * d_b).reshape(cmdp.num_states, cmdp.num_actions) + 1e-10
  pi /= np.sum(pi, axis=1, keepdims=True)
  assert np.all(pi >= -1e-6), pi

  return np.array(pi)


def cost_upper_bound(cmdp: util.CMDP,
                     w: np.ndarray,
                     d_b: np.ndarray,
                     epsilon: float):
  """Compute cost upper bound based on the DICE w.

  Args:
    cmdp: CMDP instance.
    w: stationary distribution correction estimate of the target policy.
    d_b: stationary distribution of the behavior policy.
    epsilon: hyperparameter that controls conservatism. (epsilon > 0)

  Returns:
    (cost upper bound, additional information)
  """

  if cmdp.num_costs != 1:
    raise NotImplementedError('cmdp.num_costs=1 is supported only.')
  s0 = cmdp.initial_state
  w = w.reshape(cmdp.num_states, cmdp.num_actions)
  p_n = d_b.reshape(cmdp.num_states,
                    cmdp.num_actions)[:, :, None] * cmdp.transition + 1e-10
  p_n = p_n.reshape(cmdp.num_states * cmdp.num_actions * cmdp.num_states)
  c = cmdp.costs[0, :, :]  # |S| x |A|

  def loss_fn(variables):
    tau, x = variables[0], variables[1:]
    l = (1 - cmdp.gamma) * x[s0] + w[:, :, None] * (
        c[:, :, None] + cmdp.gamma * x[None, None, :] - x[:, None, None])
    l = l.reshape(cmdp.num_states * cmdp.num_actions * cmdp.num_states)
    loss = tau * jax.nn.logsumexp(jnp.log(p_n) + l / tau) + tau * epsilon
    return loss

  loss_jit = jax.jit(loss_fn)
  grad_jit = jax.jit(jax.grad(loss_fn))

  f = lambda x: np.array(loss_jit(x))
  jac = lambda x: np.array(grad_jit(x))

  # Minimize loss_fn.
  x0 = np.ones(cmdp.num_states + 1)
  lb, ub = -np.ones_like(x0) * np.inf, np.ones_like(x0) * np.inf
  lb[0] = 0  # tau >= 0
  bounds = scipy.optimize.Bounds(lb, ub, keep_feasible=False)
  solution = scipy.optimize.minimize(
      f,
      x0=x0,
      jac=jac,
      bounds=bounds,
      options={
          'maxiter': 10000,
          'ftol': 1e-10,
          'gtol': 1e-10,
      })

  # Additional information.
  tau, x = solution.x[0], solution.x[1:]
  l = (1 - cmdp.gamma) * x[s0] + w[:, :, None] * (
      c[:, :, None] + cmdp.gamma * x[None, None, :] - x[:, None, None])
  l = l.reshape(cmdp.num_states * cmdp.num_actions * cmdp.num_states)
  loss = tau * scipy.special.logsumexp(np.log(p_n) + l / tau) + tau * epsilon
  p = scipy.special.softmax(np.log(p_n) + (l / tau)) + 1e-10
  kl = np.sum(p * np.log(p / p_n))
  p_sa = np.sum(
      p.reshape(cmdp.num_states, cmdp.num_actions, cmdp.num_states), axis=-1)
  cost_ub = np.sum(p_sa * w * c)
  info = {
      'loss': loss,
      'kl': kl,
      'cost_ub': cost_ub,
      'p': p,
      'gap': loss - cost_ub
  }

  return np.array([loss]), info


def conservative_constrained_optidice(cmdp, pi_b, alpha, epsilon, verbose=0):
  """f-divergence regularized conservative constrained RL.

  max_{d} E_d[R(s,a)] - alpha * E_{d_b}[f(d(s,a)/d_b(s,a))]
  s.t. (cost upper bound) <= hat{c}.

  We assume that f(x) = 0.5 (x-1)^2.

  Args:
    cmdp: a CMDP instance.
    pi_b: behavior policy.
    alpha: regularization hyperparameter for f-divergence.
    epsilon: degree of conservatism. (0: cost upper bound = E_d[C(s,a)]).
    verbose: whether using logging or not.

  Returns:
    the resulting policy. [num_states, num_actions]
  """
  if cmdp.num_costs != 1:
    raise NotImplementedError('cmdp.num_costs=1 is supported only.')

  lamb_left = np.array([0.0])
  lamb_right = np.array([10.0])
  start_time = time.time()

  for i in range(15):
    lamb = (lamb_left + lamb_right) * 0.5
    r_lamb = cmdp.reward - np.sum(lamb[:, None, None] * cmdp.costs, axis=0)
    mdp = util.MDP(cmdp.num_states, cmdp.num_actions, cmdp.transition, r_lamb,
                   cmdp.gamma)
    w, d_b, _ = optidice(mdp, pi_b, alpha)
    cost_mean = cmdp.costs.reshape(cmdp.num_costs, cmdp.num_states *
                                   cmdp.num_actions).dot(w * d_b)
    cost_ub, info = cost_upper_bound(cmdp, w, d_b, epsilon)
    if verbose:
      logging.info(
          '[%g] Lamb=%g, cost_ub=%.6g, gap=%.6g, kl=%.6g, cost_mean=%.6g / '
          'elapsed_time=%.3g', i, lamb, cost_ub, info['gap'], info['kl'],
          cost_mean,
          time.time() - start_time)

    if cost_ub[0] > cmdp.cost_thresholds[0]:
      lamb_left = lamb
    else:
      lamb_right = lamb

  lamb = lamb_right
  r_lamb = cmdp.reward - np.sum(lamb[:, None, None] * cmdp.costs, axis=0)
  mdp = util.MDP(cmdp.num_states, cmdp.num_actions, cmdp.transition, r_lamb,
                 cmdp.gamma)
  w, d_b, pi = optidice(mdp, pi_b, alpha)
  return pi
