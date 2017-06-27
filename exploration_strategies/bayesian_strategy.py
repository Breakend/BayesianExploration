from rllab.misc.overrides import overrides
from rllab.misc.ext import AttrDict
import rllab.misc.logger as logger
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np
import numpy.random as nr
from sampling_utils import rollout,FixedPriorityQueue
import tensorflow as tf

def kl_div_p_q(p_mean, p_std, q_mean, q_std):
    """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
    numerator = np.square(p_mean - q_mean) + \
        np.square(p_std) - np.square(q_std)
    denominator = 2 * np.square(q_std) + 1e-8
    return np.sum(numerator / denominator + np.log(q_std) - np.log(p_std))


class MCDropout(ExplorationStrategy, Serializable):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

    def __init__(self, env_spec, mu=0, theta=0.15, sigma=0.3, **kwargs):
        assert isinstance(env_spec.action_space, Box)
        assert len(env_spec.action_space.shape) == 1
        Serializable.quick_init(self, locals())
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = env_spec.action_space
        self.state = np.ones(self.action_space.flat_dim) * self.mu
        self.reset()
        self.rs = np.random.RandomState()
        self.first_mask = dict()
        self.noise_stdev = 0.9
        self.dropout_percent = .1

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["state"] = self.state
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.state = d["state"]

    @overrides
    def reset(self):
        self.state = np.ones(self.action_space.flat_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    def perturb_params(self, current_policy_params):
        # rs.randint(0, len(self.noise) - dim + 1)
        # noise_idx = noise.sample_index(rs, policy.num_params)
        v = self.noise_stdev * self.rs.randn(len(current_policy_params))
        # import pdb; pdb.set_trace
        return current_policy_params + v

    def generate_mutations(self, current_policy, num_mutations=20):
        policy_params = current_policy.get_param_values()
        mutants = []
        for i in range(num_mutations):
            perturbed_policy_params = self.perturb_params(policy_params)
            mutants.append(perturbed_policy_params)
        return mutants

    def generate_dropout_params(self, mutant_params):
        mask = np.random.binomial(size=mutant_params.shape,n=1, p=1.0 - self.dropout_percent)
        return mutant_params * mask, mask

    def get_rollout_with_dropout_mask(self, env,  current_policy, mutant_params, max_path_length, cloned_policy):
        dropped_out_params, mask = self.generate_dropout_params(mutant_params)

        cloned_policy.set_param_values(dropped_out_params)
        # import pdb; pdb.set_trace()

        return rollout(env, cloned_policy, max_path_length)

    def generate_samples(self, env, current_policy, batch_size, max_path_length=None, num_dropouts=20, acquisiton_func = "maxvar"):
        #TODO:
        # Don't need mutations
        # At every iteration run dropout 50 times
        # Will get uncertainty in the reward and action at each timestep but this uncertainty needs to propagate???
        # Take mean action, do this again. Will get uncertainty of actions and rewards of the policy at each timestep.
        # then run acquisition function
        if max_path_length is None:
            max_path_length = env.horizon

        dropped_out_policies = []
        for i in range(num_dropouts):
            params, mask = self.generate_dropout_params(current_policy.get_param_values()) 
            with tf.variable_scope("dropout_mask_%d" % i, reuse=(not self.first_mask.get(i, True))):
                cloned_policy = Serializable.clone(current_policy)
                self.first_mask[i] = False
                cloned_policy.set_param_values(params)
                dropped_out_policies.append(cloned_policy)
            
        stats = []
        print_stats = ["bayesian_return_std", "bayesian_returns_average", "kl"]
        samples_queue = FixedPriorityQueue(max_size = batch_size)
        # run for 3 times the value and only keep the most uncertain actions
        viewed_samples = 0
        while viewed_samples < batch_size*4:
            path_length = 1
            path_return = 0
            observation = env.reset()
            terminal = False
            while not terminal and path_length <= max_path_length:
                actions = []
                for cloned_policy in dropped_out_policies:
                    action, _ = cloned_policy.get_action(observation)
                    actions.append(action) 
                mean_action = np.mean(actions)
                std_action = float(np.mean(np.std(actions, axis=0)))
                assert type(std_action) is float
                next_observation, reward, terminal, _ = env.step(mean_action)
                # The heapq sorts by the first element of the tuple, thus will keep the most uncertain actions
                # we can also get the variance of the reward step by step
                try:
                    samples_queue.add((std_action, path_length, observation, mean_action, reward, terminal, path_length==1, path_length))
                except:
                    import pdb; pdb.set_trace()
                viewed_samples += 1
                observation = next_observation
                path_length += 1

        print("Average sample action variance %f" % np.mean([x[0] for x in samples_queue.heap]))

        return [x[2:] for x in samples_queue.heap]


    def get_action(self, t, observation, policy, **kwargs):
        #applying MC Dropout and taking the mean action?
        action, _ = policy.get_action(observation)
        mc_dropout = 10
        all_actions = np.zeros(shape=(mc_dropout, action.shape[0]))

        for d in range(mc_dropout):
            action, _ = policy.get_action(observation)
            all_actions[d, :] = action

        mean_action = np.mean(all_actions, axis=0)

        return mean_action



if __name__ == "__main__":
    ou = MCDropout(env_spec=AttrDict(action_space=Box(low=-1, high=1, shape=(1,))), mu=0, theta=0.15, sigma=0.3)
    states = []
    for i in range(1000):
        states.append(ou.evolve_state()[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
