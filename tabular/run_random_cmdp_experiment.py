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

"""Main experiment script."""
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np

from constrained_optidice.tabular import mdp_util
from constrained_optidice.tabular import offline_cmdp


flags.DEFINE_float('cost_thresholds', 0.1,
                   'The cost constraint threshold of the true CMDP.')
flags.DEFINE_float(
    'behavior_optimality', 0.9,
    'The optimality of data-collecting policy in terms of reward. '
    '(0: performance of uniform policy. 1: performance of optimal policy).')
flags.DEFINE_float('behavior_cost_thresholds', 0.1,
                   'Set the cost value of data-collecting policy.')
flags.DEFINE_integer('num_iterations', 10,
                     'The number of iterations for the repeated experiments.')

FLAGS = flags.FLAGS


def main(unused_argv):
  """Main function."""
  num_states, num_actions, num_costs, gamma = 50, 4, 1, 0.95
  cost_thresholds = np.ones(num_costs) * FLAGS.cost_thresholds
  behavior_optimality = FLAGS.behavior_optimality
  behavior_cost_thresholds = np.array([FLAGS.behavior_cost_thresholds])

  logging.info('==============================')
  logging.info('Cost threshold: %g', cost_thresholds)
  logging.info('Behavior optimality: %g', behavior_optimality)
  logging.info('Behavior cost thresholds: %g', behavior_cost_thresholds)
  logging.info('==============================')

  results = []
  start_time = time.time()
  for seed in range(FLAGS.num_iterations):
    # Construct a random CMDP
    np.random.seed(seed)
    cmdp = mdp_util.generate_random_cmdp(num_states, num_actions, num_costs,
                                         cost_thresholds, gamma)

    result = {}
    # Optimal policy for unconstrained MDP
    pi_uopt, _, _ = mdp_util.solve_mdp(cmdp)
    v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_uopt)
    uopt_r, uopt_c = v_r[0], v_c[0][0]
    result.update({'uopt_r': uopt_r, 'uopt_c': uopt_c})

    # Optimal policy for constrained MDP
    pi_copt = mdp_util.solve_cmdp(cmdp)
    v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_copt)
    opt_r, opt_c = v_r[0], v_c[0][0]
    result.update({'opt_r': opt_r, 'opt_c': opt_c})

    # Construct behavior policy
    pi_b = offline_cmdp.generate_baseline_policy(
        cmdp,
        behavior_cost_thresholds=behavior_cost_thresholds,
        optimality=behavior_optimality)
    v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_b)
    pib_r, pib_c = v_r[0], v_c[0][0]
    result.update({'behav_r': pib_r, 'behav_c': pib_c})

    for num_trajectories in [10, 20, 50, 100, 200, 500, 1000, 2000]:
      logging.info('==========================')
      logging.info('* seed=%d, num_trajectories=%d', seed, num_trajectories)
      alpha = 1. / num_trajectories  # Parameter for CCIDCE, CDICE.

      # Generate trajectory
      trajectory = mdp_util.generate_trajectory(
          seed, cmdp, pi_b, num_episodes=num_trajectories)

      # MLE CMDP
      mle_cmdp, _ = mdp_util.compute_mle_cmdp(num_states, num_actions,
                                              num_costs, cmdp.reward,
                                              cmdp.costs, cost_thresholds,
                                              gamma, trajectory)

      # Basic RL
      pi = mdp_util.solve_cmdp(mle_cmdp)
      v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
      basic_r = v_r[0]
      basic_c = v_c[0][0]
      result.update({'basic_r': basic_r, 'basic_c': basic_c})

      # Vanilla ConstrainedOptiDICE
      pi = offline_cmdp.constrained_optidice(mle_cmdp, pi_b, alpha)
      v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
      cdice_r = v_r[0]
      cdice_c = v_c[0][0]
      result.update({'cdice_r': cdice_r, 'cdice_c': cdice_c})

      # Conservative ConstrainedOptiDICE
      epsilon = 0.1 / num_trajectories
      pi = offline_cmdp.conservative_constrained_optidice(
          mle_cmdp, pi_b, alpha=alpha, epsilon=epsilon)
      v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
      ccdice_r = v_r[0]
      ccdice_c = v_c[0][0]
      result.update({'ccdice_r': ccdice_r, 'ccdice_c': ccdice_c})

      # Print the result
      elapsed_time = time.time() - start_time
      result.update({
          'seed': seed,
          'num_trajectories': num_trajectories,
          'elapsed_time': elapsed_time,
      })
      logging.info(result)
      results.append(result)


if __name__ == '__main__':
  app.run(main)
