import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class BaseModel(tf.keras.Model):
    def __init__(self, name, input_size = 1):
        super().__init__(name)
        self.input_size = input_size
        self.dist = ProbabilityDistribution()
    
    def reset_buffer(self, initial_obs):
        self.buffer = np.array([initial_obs.copy() for _ in range(self.input_size)])

    def action_value(self, obs, training = False):
        # If action_value is called for prediction on
        # subsequent observations
        if not training and self.input_size > 1:
            np.roll(self.buffer,-1)
            self.buffer[-1] = obs
            obs = self.buffer
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class CNNModel(BaseModel):
    def __init__(self, num_actions):
        super().__init__('CNN')
        # Note: no tf.get_variable(), just simple Keras API!
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # Logits are unnormalized log probabilities.
        self.logits = kl.Dense(num_actions, name='policy_logits')

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

class SimpleModel(BaseModel):
    def __init__(self, num_actions):
        super().__init__('Simple')
        # Note: no tf.get_variable(), just simple Keras API!
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # Logits are unnormalized log probabilities.
        self.logits = kl.Dense(num_actions, name='policy_logits')

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)