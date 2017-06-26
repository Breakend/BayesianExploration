from rllab.misc.overrides import overrides
from rllab.misc.ext import AttrDict
import rllab.misc.logger as logger
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np
import numpy.random as nr
from sampling_utils import rollout
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
        self.first_mask = True
        self.noise_stdev = 0.02
        self.dropout_percent = .05

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

    def generate_mutations(self, current_policy, num_mutations=10):
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

    def generate_samples(self, env, current_policy, batch_size, max_path_length=None, num_dropouts=20, acquisiton_func = "kl"):
        if max_path_length is None:
            max_path_length = env.horizon

        with tf.variable_scope("dropout_mask", reuse=not self.first_mask):
            cloned_policy = Serializable.clone(current_policy)
            self.first_mask = False

        # generate your sampled parameter space
        mutants = self.generate_mutations(current_policy)
        current_rollouts = []
        current_policy_params = current_policy.get_param_values()

        for i in range(num_dropouts):
            current_rollouts.append(self.get_rollout_with_dropout_mask(env, current_policy, current_policy_params, max_path_length, cloned_policy))
        rewards = [np.sum(step[2]) for r in current_rollouts for step in r]
        current_mean = np.mean(rewards)
        current_var = np.var(rewards)

        stats = []
        print_stats = ["bayesian_return_variance", "bayesian_returns_average", "kl"]

        for mutant in mutants:
            # print("Running mutant")
            rollouts = []
            # Run pseudo-mcdropout and then check the variance over the reward
            # TODO: do we need baseline over the variance of the parameters themselves from dropout?
            for i in range(num_dropouts):
                # print("Running dropout %d" % i)
                rollouts.append(self.get_rollout_with_dropout_mask(env, current_policy, mutant, max_path_length, cloned_policy))
            # import pdb; pdb.set_trace()
            rewards = [np.sum(step[2]) for r in rollouts for step in r]
            variance = np.var(rewards)
            mean = np.mean(rewards)
            stat = dict(mutant=mutant, bayesian_return_variance=variance, bayesian_returns_average=mean, kl=kl_div_p_q(current_mean, current_var, mean, variance), rollouts=rollouts)
            for s in print_stats:
                logger.log(s + " : " + str(stat[s]))
            stats.append(stat)

        #TODO: extract to util funcs 
        if acquisiton_func == "kl":
            stat = max(stats, key=(lambda x: x["kl"]))
            params = stat["mutant"]

        cloned_policy.set_param_values(params)

        # Use the dropout rollouts
        samples = [step for rollout in stat["rollouts"] for step in rollout]

        while len(samples) < batch_size:
            samples.extend(rollout(env, cloned_policy, max_path_length))
        return samples


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
