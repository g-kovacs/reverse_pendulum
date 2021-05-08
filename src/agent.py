import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import numpy as np
import collections

class A2CAgent:
    def __init__(self, lr=7e-3, gamma=0.999, value_c=0.5, entropy_c=1e-4):
        # `gamma` is the discount factor
        self.gamma = gamma
        # Coefficients are used for the loss terms.
        self.value_c = value_c
        self.entropy_c = entropy_c
        self.lr = lr

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kls.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss

    def train(self, env, models, batch_size=128, updates=500):
        if not isinstance(models, (collections.Sequence, np.ndarray)):
            models = np.array([models])
        for model in models:
            model.compile(
                optimizer=ko.RMSprop(lr=self.lr),
                # Define separate losses for policy logits and value estimate.
                loss=[self._logits_loss, self._value_loss])
        model_num = models.shape[0]
        # Storage helpers for a single batch of data.
        actions = np.empty((model_num, batch_size), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        max_window_size = 0
        for model in models:
            if(max_window_size < model.window_size):
                max_window_size = model.window_size
        model = models[0]    #TODO remove
        observations = np.empty((batch_size, ) + env.observation_space.shape)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = []
        next_obs = env.reset()
        model.reset_buffer(next_obs)
        next_reward = 0.0
        for _ in range(updates):
            for step in range(batch_size):
                actions[step], values[step] = model.action_value(next_obs, False)
                if model.window_size > 1:
                    observations[step] = model.buffer
                else:
                    observations[step] = next_obs
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                next_reward += rewards[step]
                if dones[step]:
                    ep_rewards.append(next_reward)
                    next_reward = 0
                    next_obs = env.reset()
                    if model.window_size > 1:
                        model.reset_buffer(next_obs)

            _, next_value = model.action_value(next_obs, False)

            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

            model.train_on_batch(observations, [acts_and_advs, returns])
        return ep_rewards

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values

        return returns, advantages
