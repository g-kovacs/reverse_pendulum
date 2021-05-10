import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np
import collections
import os
from shutil import make_archive, rmtree, unpack_archive
from ast import literal_eval
from DCPEnv import DCPEnv


class ModelConfiguration:
    @property
    def window_size(self):
        max_window = 1
        for m in self.__models:
            if m.input_size > max_window:
                max_window = m.input_size
        return max_window

    @property
    def num(self):
        return len(self.__models)

    def __init__(self, models, label=None):
        
        if not isinstance(models, (collections.Sequence, np.ndarray)):
            models = [models]
        if label is None:
            self.label = 'vs'.join([model.label for model in models])
        else:
            self.label = label
        self.__models = models

    def get(self):
        return self.__models

    def clean(self):
        rmtree(os.path.join('saves', self.label))

    def save(self):
        configs = []
        model_names = []
        dict_path = os.path.join('saves', self.label)
        if not os.path.exists(dict_path):
            os.makedirs(dict_path)
        for file in os.scandir(dict_path):
            if file.is_dir():
                rmtree(file)
            if file.is_file():
                os.remove(file)
        for idx, model in enumerate(self.__models):
            name = '-'.join((str(idx), model.label))
            save_path = os.path.join(dict_path, name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_weights(os.path.join(save_path, 'save_data'))
            configs.append(str(model.get_config()))
            model_names.append(name)
        with open(os.path.join(dict_path, 'config'), 'w') as cfg_file:
            for config in configs:
                cfg_file.write(config + '\n')
        with open(os.path.join(dict_path, 'names'), 'w') as name_file:
            for name in model_names:
                name_file.write(name + '\n')
        os.chdir("saves")
        make_archive(self.label, 'zip', base_dir=self.label)
        rmtree(self.label)
        os.chdir('..')

    @classmethod
    def load(cls, cfg_name='default'):
        models = []
        os.chdir('saves')
        unpack_archive(filename='.'.join((cfg_name, 'zip')), format='zip')
        cfg_path = os.path.join(cfg_name, 'config')
        names_path = os.path.join(cfg_name, 'names')
        with open(cfg_path, 'r') as cfg_file:
            with open(names_path, 'r') as name_file:
                for params, name in zip(cfg_file, name_file):
                    d = literal_eval(params)
                    cls_ = getattr(__import__(__name__), d['class'])
                    m = cls_(*tuple([DCPEnv.actions_size, *d.values()]))
                    m.compile()
                    m.load_weights(os.path.join(os.getcwd(),
                                    cfg_name,
                                    name.strip('\n'),
                                    'save_data')).expect_partial()
                    models.append(m)
        os.chdir("..")
        return cls(models, cfg_name)


class BaseModel(tf.keras.Model):
    class ProbabilityDistribution(tf.keras.Model):
        def call(self, logits, **kwargs):
            return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

    labels = {}

    def __init__(self, name, input_size=1):
        super().__init__(name)
        if name not in BaseModel.labels:
            BaseModel.labels[name] = 0
        BaseModel.labels[name] += 1
        self.label = name + '_' + str(BaseModel.labels[name])
        self.input_size = input_size
        self.dist = BaseModel.ProbabilityDistribution()

    def action_value(self, obs):
        obs = obs[-self.input_size:]
        if self.input_size > 1:
            obs = obs[None, :]
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        # TODO test action shapes
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def get_config(self):
        return {'class': self.__class__.__name__, '*size': self.input_size}


class CNNModel(BaseModel):
    def __init__(self, num_actions, name='CNNModel', memory_size=8, *args):
        super().__init__(name, memory_size)
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
    def __init__(self, num_actions, name='LSTMModel', memory_size=8, *args):
        print('lstm: ', num_actions, name, memory_size, args)
        super().__init__(name, memory_size)
        self.lstm = kl.LSTM(16)
        self.actor = kl.Dense(64, activation='relu', kernel_initializer='he_normal')
        self.critic = kl.Dense(64, activation='relu', kernel_initializer='he_normal')

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
    def __init__(self, num_actions, name='SimpleAC2', *args):
        super().__init__(name)

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
    def __init__(self, num_actions, name='SimpleAC', *args):
        super().__init__(name)

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
