from rllab.misc.overrides import overrides
from rllab.misc.ext import AttrDict
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np
import numpy.random as nr
from sampling_utils import rollout
import tensorflow as tf

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

    def generate_mutations(self, current_policy, num_mutations=4):
        policy_params = current_policy.get_param_values()
        mutants = []
        for i in range(num_mutations):
            perturbed_policy_params = self.perturb_params(policy_params)
            mutants.append(perturbed_policy_params)
        return mutants

    def generate_dropout_params(self, mutant_params):
        mask = np.random.binomial(mutant_params.shape, 1.0-self.dropout_percent)[0] * (1.0/(1-self.dropout_percent))
        return mutant_params * mask, mask

    def get_rollout_with_dropout_mask(self, env,  current_policy, mutant_params, max_path_length, cloned_policy):
        dropped_out_params, mask = self.generate_dropout_params(mutant_params)

        cloned_policy.set_param_values(dropped_out_params)
        # import pdb; pdb.set_trace()

        return rollout(env, cloned_policy, max_path_length)

    def generate_samples(self, env, current_policy, batch_size, max_path_length=None, num_dropouts=20):
        if max_path_length is None:
            max_path_length = env.horizon

        with tf.variable_scope("dropout_mask", reuse=not self.first_mask):
            cloned_policy = Serializable.clone(current_policy)
            self.first_mask = False

        # generate your sampled parameter space
        mutants = self.generate_mutations(current_policy)

        highest_variance_pair = None

        variances = []

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
            variances.append(variance)
            mean = np.mean(rewards)
            # TODO: make this much more abstract, like keep it in a table of info and then run an acquisition over the info
            if highest_variance_pair is None or highest_variance_pair[1] < variance:
                highest_variance_pair = (mutant, variance)
        print("----variances---")
        print(variances)
        print("----variances---")
        with tf.variable_scope("dropout_mask", reuse=not self.first_mask):
            cloned_policy = Serializable.clone(current_policy)
            cloned_policy.set_param_values(highest_variance_pair[0])
        samples = []
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
