import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import numpy as np
import collections

class A2CAgent:
    def __init__(self, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
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

    def train(self, env, config, batch_size=128, updates=500, max_step = 300):
        models = config.get()
        for model in models:
            model.compile(optimizer=ko.RMSprop(lr=self.lr),
                    loss=[self._logits_loss, self._value_loss])
        model_num = len(models)
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_size, config.num), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size, config.num))
        observations = np.empty((batch_size, config.window_size, env.observations_size))

        # Training loop: collect samples, send to optimizer, repeat updates times.
        deaths = {}
        for model in models:
            deaths[model.label] = 0
        obs_window = env.reset()
        episodes = []
        steps = 0
        for _ in range(updates):
            for step in range(batch_size):
                steps += 1
                observations[step] = obs_window
                for m_i, model in enumerate(models):
                    actions[step, m_i], values[step, m_i] = model.action_value(obs_window)
                obs_window, rewards[step], dones[step] = env.step(actions[step])
                if any(dones[step]) or max_step < steps:
                    obs_window = env.reset()
                    episodes.append(steps*env.dt)
                    steps = 0
                    for dead, model in zip(dones[step], models):
                        if dead:
                            deaths[model.label] += 1
            for m_i, model in enumerate(models):
                _, next_value = model.action_value(obs_window)
                returns, advs = self._returns_advantages(rewards[:,m_i],
                                                        dones[:,m_i],
                                                        values[:,m_i],
                                                        next_value)
                # A trick to input actions and advantages through same API.
                acts_and_advs = np.concatenate([actions[:,m_i, None], advs[:, None]], axis=-1)
                model.train_on_batch(observations[:,-model.input_size:,:], [acts_and_advs, returns])
        return episodes, deaths

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values

        return returns, advantages
