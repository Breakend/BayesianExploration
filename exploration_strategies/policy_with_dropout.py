from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.misc import tensor_utils

import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.layers import batch_norm

from sandbox.rocky.tf.spaces.discrete import Discrete
import tensorflow as tf

import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import numpy as np
import itertools
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.core.layers_powered import LayersPowered


class DropoutMLP(LayersPowered, Serializable):
    def __init__(self, name, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, dropout_prob=.1, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer(),
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer(),
                 input_var=None, input_layer=None, input_shape=None, batch_normalization=False, weight_normalization=False,
                 ):

        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                l_hid = L.batch_norm(l_hid)
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_normalization=weight_normalization
                )
                l_hid = L.DropoutLayer(l_hid, dropout_prob) 
                self._layers.append(l_hid)
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )
            if batch_normalization:
                l_out = L.batch_norm(l_out)
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out
            # self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class DeterministicDropoutMLPPolicy(Policy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            prob_network=None,
            bn=False):
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            if prob_network is None:
                prob_network = DropoutMLP(
                    input_shape=(env_spec.observation_space.flat_dim,),
                    output_dim=env_spec.action_space.flat_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    # batch_normalization=True,
                    name="prob_network",
                )

            self._l_prob = prob_network.output_layer
            self._l_obs = prob_network.input_layer
            self._f_prob = tensor_utils.compile_function(
                [prob_network.input_layer.input_var],
                L.get_output(prob_network.output_layer, deterministic=True)
            )

            self._f_prob_dropout = tensor_utils.compile_function(
                [prob_network.input_layer.input_var],
                L.get_output(prob_network.output_layer, deterministic=False)
            )

        self.prob_network = prob_network

        # Note the deterministic=True argument. It makes sure that when getting
        # actions from single observations, we do not update params in the
        # batch normalization layers.
        # TODO: this doesn't currently work properly in the tf version so we leave out batch_norm
        super(DeterministicDropoutMLPPolicy, self).__init__(env_spec)
        LayersPowered.__init__(self, [prob_network.output_layer])

    def get_action_with_dropout(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        action = self._f_prob_dropout([flat_obs])[0]
        return action, dict()

    @property
    def vectorized(self):
        return True
        
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        action = self._f_prob([flat_obs])[0]
        return action, dict()

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        actions = self._f_prob(flat_obs)
        return actions, dict()

    def get_action_sym(self, obs_var):
        return L.get_output(self.prob_network.output_layer, obs_var)
