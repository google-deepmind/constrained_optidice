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

"""Utility functions for tabular MDPs and CMDPs."""
from typing import Tuple

from absl import logging
import cvxopt
import jax
import jax.config
import numpy as np
import scipy
import scipy.optimize


cvxopt.solvers.options['show_progress'] = False
jax.config.update('jax_enable_x64', True)


class MDP:
  """MDP class."""

  def __init__(self,
               num_states: int,
               num_actions: int,
               transition: np.ndarray,
               reward: np.ndarray,
               gamma: float):
    """MDP Constructor.

    Args:
      num_states: the number of states.
      num_actions: the number of actions.
      transition: transition matrix. [num_states, num_actions, num_states].
      reward: reward function. [num_states, num_actions]
      gamma: discount factor (0 ~ 1).
    """
    self.num_states = num_states
    self.num_actions = num_actions
    self.transition = np.array(transition)
    self.reward = np.array(reward)
    self.gamma = gamma
    self.initial_state = 0
    self.absorbing_state = num_states - 1
    assert self.transition.shape == (num_states, num_actions, num_states)
    assert self.reward.shape == (num_states, num_actions)

  def __copy__(self):
    mdp = MDP(
        num_states=self.num_states,
        num_actions=self.num_actions,
        transition=np.array(self.transition),
        reward=np.array(self.reward),
        gamma=self.gamma)
    return mdp


class CMDP(MDP):
  """Constrained MDP class."""

  def __init__(self,
               num_states: int,
               num_actions: int,
               num_costs: int,
               transition: np.ndarray,
               reward: np.ndarray,
               costs: np.ndarray,
               cost_thresholds: np.ndarray,
               gamma: float):
    """Constrained MDP Constructor.

    Args:
      num_states: the number of states.
      num_actions: the number of actions.
      num_costs: the number of cost types.
      transition: transition matrix. [num_states, num_actions, num_states].
      reward: reward function. [num_states, num_actions]
      costs: cost function. [num_costs, num_states, num_actions]
      cost_thresholds: cost thresholds. [num_costs]
      gamma: discount factor (0 ~ 1).
    """
    assert len(cost_thresholds) == num_costs
    super(CMDP, self).__init__(num_states, num_actions, transition, reward,
                               gamma)
    self.num_costs = num_costs
    self.costs = np.array(costs)
    self.cost_thresholds = np.array(cost_thresholds)
    assert self.costs.shape == (num_costs, num_states, num_actions)

  def __copy__(self):
    cmdp = CMDP(
        num_states=self.num_states,
        num_actions=self.num_actions,
        num_costs=self.num_costs,
        transition=np.array(self.transition),
        reward=np.array(self.reward),
        costs=np.array(self.costs),
        cost_thresholds=np.array(self.cost_thresholds),
        gamma=self.gamma)
    return cmdp


def generate_random_cmdp(num_states: int, num_actions: int, num_costs: int,
                         cost_thresholds: np.ndarray, gamma: float):
  """Create a random CMDP.

  Args:
    num_states: the number of states.
    num_actions: the number of actions.
    num_costs: the number of cost types.
    cost_thresholds: cost thresholds. [num_costs]
    gamma: discount factor (0 ~ 1).

  Returns:
    a CMDP instance.
  """
  assert len(cost_thresholds) == num_costs
  if num_costs != 1:
    raise NotImplementedError('Only support when num_costs=1')
  initial_state = 0
  absorbing_state = num_states - 1  # the absorbing state index.

  # Define a random transition.
  transition = np.zeros((num_states, num_actions, num_states))
  for s in range(num_states):
    if s == absorbing_state:
      transition[s, :, s] = 1  # absorbing state: self-transition
    else:
      for a in range(num_actions):
        # Transition to next states is defined sparsely.
        # For each (s,a), the connectivity to the next states is 4.
        p = np.r_[np.random.dirichlet([1, 1, 1, 1]), [0] * (num_states - 4 - 1)]
        np.random.shuffle(p)
        transition[s, a, :] = np.r_[p, [0]]

  # Define a reward function. Roughly speaking, a non-zero reward is given
  # to the state which is most difficult to reach from the initial state.
  min_value_state, min_value = -1, 1e10
  for s in range(num_states - 1):
    reward = np.zeros((num_states, num_actions))
    reward[s, :] = 1 / (1 - gamma)
    transition_tmp = np.array(transition[s, :, :])
    transition[s, :, :] = 0
    transition[s, :, absorbing_state] = 1  # from goal_state to absorbing state
    mdp = MDP(num_states, num_actions, transition, reward, gamma)
    _, v, _ = solve_mdp(mdp)
    if v[initial_state] < min_value:
      min_value = v[initial_state]
      min_value_state = s
    transition[s, :, :] = transition_tmp
  # min_value_state will be the goal state that yields a non-zero reward.
  goal_state = min_value_state
  reward = np.zeros((num_states, num_actions))
  reward[goal_state, :] = 1 / (1 - gamma)
  transition[goal_state, :, :] = 0
  transition[goal_state, :, absorbing_state] = 1  # to absorbing one

  # Define a cost function.
  while True:
    costs = np.random.beta(0.2, 0.2, (num_costs, num_states, num_actions))
    # For each state, there exists a no-cost action.
    for s in range(num_states):
      a_no_cost = np.random.randint(0, num_actions)
      costs[:, s, a_no_cost] = 0
    costs[:, absorbing_state, :] = 0
    cmdp = CMDP(num_states, num_actions, num_costs, transition, reward, costs,
                cost_thresholds, gamma)
    pi_copt = solve_cmdp(cmdp)
    v_c_opt = policy_evaluation(cmdp, pi_copt)[2][0, 0]
    if v_c_opt >= cost_thresholds[0] - 1e-4:
      # We want that optimal policy tightly matches the cost constraint.
      break

  cmdp = CMDP(num_states, num_actions, num_costs, transition, reward, costs,
              cost_thresholds, gamma)
  return cmdp


def policy_evaluation_mdp(mdp: MDP,
                          pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Policy evaluation (normalized value) for pi in the given MDP.

  Args:
    mdp: MDP instance.
    pi: a stochastic policy. [num_states, num_actions]

  Returns:
    (V_R, Q_R)
  """
  reward = mdp.reward * (1 - mdp.gamma)  # normalized value
  r = np.sum(mdp.reward * pi, axis=-1)
  p = np.sum(pi[:, :, None] * mdp.transition, axis=1)
  v = np.linalg.inv(np.eye(mdp.num_states) - mdp.gamma * p).dot(r)
  q = reward + mdp.gamma * mdp.transition.dot(v)
  return v, q


def policy_evaluation(
    cmdp: CMDP,
    pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Policy evaluation (normalized value) for pi in the given CMDP.

  Args:
    cmdp: CMDP instance.
    pi: a stochastic policy. [num_states, num_actions]

  Returns:
    (V_R, Q_R, V_C, Q_C)
  """
  def compute(transition, reward, pi):
    reward = reward * (1 - cmdp.gamma)  # normalized value
    r = np.sum(reward * pi, axis=-1)
    p = np.sum(pi[:, :, None] * transition, axis=1)
    v = np.linalg.inv(np.eye(cmdp.num_states) - cmdp.gamma * p).dot(r)
    q = reward + cmdp.gamma * cmdp.transition.dot(v)
    return v, q

  v_r, q_r = compute(cmdp.transition, cmdp.reward, pi)
  v_cs = np.zeros((cmdp.num_costs, cmdp.num_states))
  q_cs = np.zeros((cmdp.num_costs, cmdp.num_states, cmdp.num_actions))
  for k in range(cmdp.num_costs):
    v_c, q_c = compute(cmdp.transition, cmdp.costs[k], pi)
    v_cs[k] = v_c
    q_cs[k] = q_c
  return v_r, q_r, v_cs, q_cs


def solve_mdp(mdp: MDP):
  """Solve MDP via policy iteration.

  Args:
    mdp: an MDP instance.

  Returns:
    (pi, V_R, Q_R).
  """
  pi = np.ones((mdp.num_states, mdp.num_actions)) / mdp.num_actions
  v_old = np.zeros(mdp.num_states)

  for _ in range(1_000_000):
    v, q = policy_evaluation_mdp(mdp, pi)
    pi_new = np.zeros((mdp.num_states, mdp.num_actions))
    pi_new[np.arange(mdp.num_states), np.argmax(q, axis=1)] = 1.

    if np.all(pi == pi_new) or np.max(np.abs(v - v_old)) < 1e-8:
      break
    v_old = v
    pi = pi_new

  if not np.all(pi == pi_new):
    logging.info('Warning: policy iteration process did not converge.')
  return pi, v, q


def generate_trajectory(seed: int,
                        cmdp: CMDP,
                        pi: np.ndarray,
                        num_episodes: int = 10,
                        max_timesteps: int = 50):
  """Generate trajectories using the policy in the CMDP.

  Args:
    seed: random seed.
    cmdp: CMDP instance.
    pi: a stochastic policy. [num_states, num_actions]
    num_episodes: the number of episodes to generate.
    max_timesteps: the maximum timestep within an episode.

  Returns:
    trajectory: list of list of (episode_idx, t, s_t, a_t, r_t, c_t, s_t').
  """
  if seed is not None:
    np.random.seed(seed + 1)

  def random_choice_prob_vectorized(p):
    """Batch random_choice.

    e.g. p = np.array([
        [0.1, 0.5, 0.4],
        [0.8, 0.1, 0.1]])

    Args:
      p: batch of probability vector.

    Returns:
      Sampled indices
    """
    r = np.expand_dims(np.random.rand(p.shape[0]), axis=1)
    return (p.cumsum(axis=1) > r).argmax(axis=1)

  trajectory = [[] for i in range(num_episodes)]
  done = np.zeros(num_episodes, dtype=np.bool)
  state = np.array([cmdp.initial_state] * num_episodes)
  for t in range(max_timesteps):
    action = random_choice_prob_vectorized(p=pi[state, :])
    reward = cmdp.reward[state, action]
    costs = cmdp.costs[:, state, action]
    state1 = random_choice_prob_vectorized(p=cmdp.transition[state, action, :])
    for i in range(num_episodes):
      if not done[i]:
        trajectory[i].append(
            (i, t, state[i], action[i], reward[i], costs[:, i], state1[i]))
    done = done | (state == cmdp.absorbing_state)

    state = state1

  return trajectory


def compute_mle_cmdp(num_states: int,
                     num_actions: int,
                     num_costs: int,
                     reward: np.ndarray,
                     costs: np.ndarray,
                     cost_thresholds: np.ndarray,
                     gamma: float,
                     trajectory,
                     absorb_unseen: bool = True) -> Tuple[CMDP, np.ndarray]:
  """Construct a maximum-likelihood estimation CMDP from the trajectories.

  Args:
    num_states: the number of states.
    num_actions: the number of actions.
    num_costs: the number of costs.
    reward: reward function.
    costs: cost function.
    cost_thresholds: cost thresholds.
    gamma: discount factor (0~1).
    trajectory: trajectories collected by a behavior policy.
      list of list of (episode_idx, t, s_t, a_t, r_t, c_t, s_t').
    absorb_unseen: for unvisited s, whether to use transition to absorbing
      state. If False, uniform transition is used.

  Returns:
    (MLE CMDP, visitation count matrix)
  """
  absorbing_state = num_states - 1
  n = np.zeros((num_states, num_actions, num_states))
  for trajectory_one in trajectory:
    # episode, t, s, a, r, c, s1
    for _, _, s, a, _, _, s1 in trajectory_one:
      n[s, a, s1] += 1

  transition = np.zeros((num_states, num_actions, num_states))
  for s in range(num_states):
    for a in range(num_actions):
      if n[s, a, :].sum() == 0:
        if absorb_unseen:
          transition[s, a, absorbing_state] = 1  # absorbing state
        else:
          transition[s, a, :] = 1. / num_states
      else:
        transition[s, a, :] = n[s, a, :] / n[s, a, :].sum()

  mle_cmdp = CMDP(num_states, num_actions, num_costs, transition, reward, costs,
                  cost_thresholds, gamma)

  return mle_cmdp, n


def solve_cmdp(cmdp: CMDP):
  """Solve CMDP via linear programming.

  Args:
    cmdp: a CMDP instance.

  Returns:
    optimal policy.
  """
  c = -cmdp.reward.reshape(cmdp.num_states * cmdp.num_actions)
  p0 = np.zeros(cmdp.num_states)
  p0[cmdp.initial_state] = 1
  p = cmdp.transition.reshape(cmdp.num_states * cmdp.num_actions,
                              cmdp.num_states)  # |S||A| x |S|
  p = p / np.sum(p, axis=1, keepdims=True)
  b = np.repeat(
      np.eye(cmdp.num_states), cmdp.num_actions, axis=0)  # |S||A| x |S|

  a_eq = (b - cmdp.gamma * p).T
  b_eq = (1 - cmdp.gamma) * p0
  a_ub = cmdp.costs.reshape(cmdp.num_costs, cmdp.num_states * cmdp.num_actions)
  b_ub = cmdp.cost_thresholds

  #  Minimize::
  #     c @ x
  # Subject to::
  #     A_ub @ x <= b_ub
  #     A_eq @ x == b_eq
  #      lb <= x <= ub
  # where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.
  res = scipy.optimize.linprog(
      c,
      A_ub=a_ub,
      b_ub=b_ub,
      A_eq=a_eq,
      b_eq=b_eq,
      bounds=(0, np.inf),
      options={
          'maxiter': 10000,
          'tol': 1e-8
      })
  assert np.all(res.x >= -1e-4)

  d = np.clip(res.x.reshape(cmdp.num_states, cmdp.num_actions), 1e-10, np.inf)
  pi = d / np.sum(d, axis=1, keepdims=True)
  return pi


