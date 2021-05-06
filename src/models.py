import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class BaseModel(tf.keras.Model):
    def __init__(self, name, window_size = 1):
        super().__init__(name)
        self.label = name
        self.window_size = window_size
        self.dist = ProbabilityDistribution()
    
    def reset_buffer(self, initial_obs):
        self.buffer = np.array([initial_obs.copy() for _ in range(self.window_size)])

    def action_value(self, obs, training = True):
        # If called for prediction on
        # subsequent observations
        if not training and self.window_size > 1:
            np.roll(self.buffer,-1)
            self.buffer[-1] = obs[0]
            obs = self.buffer[None,:]
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class CNNModel(BaseModel):
    def __init__(self, num_actions, memory_size=8):
        super().__init__('CNNModel', memory_size)
        self.cnn = kl.Conv1D(filters=2, kernel_size=4)
        self.norm = kl.BatchNormalization()
        self.activation = kl.ReLU()
        self.flatten = kl.Flatten()
        self.actor = kl.Dense(64, activation='relu', kernel_initializer='he_normal')
        self.critic = kl.Dense(64, activation='relu', kernel_initializer='he_normal')

        self.value = kl.Dense(1, name='value')
        self.logits = kl.Dense(num_actions, name='policy_logits')

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)
        # Decoder
        cnn = self.cnn(x)
        norm = self.norm(cnn)
        act = self.activation(norm)
        features = self.flatten(act)
        # Actor-Critic
        hidden_logits = self.actor(features)
        hidden_values = self.critic(features)
        return self.logits(hidden_logits), self.value(hidden_values)

class LSTMModel(BaseModel):
    def __init__(self, num_actions,memory_size=8):
        super().__init__('LSTMModel', memory_size)
        self.lstm = kl.LSTM(64)
        self.actor = kl.Dense(32, activation='relu', kernel_initializer='he_normal')
        self.critic = kl.Dense(32, activation='relu', kernel_initializer='he_normal')

        self.value = kl.Dense(1, name='value')
        self.logits = kl.Dense(num_actions, name='policy_logits')

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)
        # Decoder
        features = self.lstm(x)
        # Actor-Critic
        hidden_logits = self.actor(features)
        hidden_values = self.critic(features)
        return self.logits(hidden_logits), self.value(hidden_values)

class SimpleAC2(BaseModel):
    def __init__(self, num_actions):
        super().__init__('SimpleAC2')

        self.actor = kl.Dense(128, activation='relu', kernel_initializer='he_normal')
        self.critic = kl.Dense(128, activation='relu', kernel_initializer='he_normal')

        self.value = kl.Dense(1, name='value')
        self.logits = kl.Dense(num_actions, name='policy_logits')

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)
        
        hidden_logits = self.actor(x)
        hidden_values = self.critic(x)
        return self.logits(hidden_logits), self.value(hidden_values)

class SimpleAC(BaseModel):
    def __init__(self, num_actions):
        super().__init__('SimpleAC')
        
        self.decoder = kl.Dense(64, activation='relu', kernel_initializer='he_normal')
        self.actor = kl.Dense(32, activation='relu', kernel_initializer='he_normal')
        self.critic = kl.Dense(32, activation='relu', kernel_initializer='he_normal')

        self.value = kl.Dense(1, name='value')
        self.logits = kl.Dense(num_actions, name='policy_logits')

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)
        
        features = self.decoder(x)
        hidden_logits = self.actor(features)
        hidden_values = self.critic(features)
        return self.logits(hidden_logits), self.value(hidden_values)