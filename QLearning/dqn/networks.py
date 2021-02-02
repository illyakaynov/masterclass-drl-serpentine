import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from dqn import NoisyDense


class DeepQNetwork:
    def __init__(self, state_shape, n_actions, learning_rate=1e-4, gamma=0.99):
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.online_network = self.create_model('dqn')

        self.state_shape = state_shape

        self.gamma = tf.constant(gamma)

        self.loss_layer = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def create_model(self, model_name):
        input_state = tf.keras.Input(shape=self.state_shape,
                                     batch_size=None,
                                     name='state_input',
                                     dtype=tf.uint8)

        x = tf.divide(tf.cast(input_state, dtype=tf.float32), tf.constant(255., dtype=tf.float32))

        x = Conv2D(16, kernel_size=(8, 8), strides=(4, 4),
                   padding='valid', activation='relu', name='conv1')(x)
        x = Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                   padding='valid', activation='relu', name='conv2')(x)
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                   padding='valid', activation='relu', name='conv3')(x)

        x = Flatten()(x)

        x = Dense(256, activation='tanh', name='hidden_dense')(x)

        x = Dense(512, activation='tanh', name='hidden_dense_value')(x)

        q_values = Dense(self.n_actions, activation='linear', name='value_output')(x)

        q_network = Model(inputs=[input_state], outputs=[q_values], name=model_name)
        return q_network

    @tf.function
    def get_best_action(self, state):
        return tf.argmax(self.online_network(state), axis=1)

    @tf.function
    def get_best_action_value(self, state):
        return tf.reduce_max(self.online_network(state), axis=1)

    @tf.function
    def train_op(self, replay_state, replay_action, replay_rewards, replay_next_state, terminal):
        replay_continues = 1.0 - terminal
        # get max q-values for the next state
        q_best_next = self.get_best_action_value(replay_next_state)
        # calculate target, if episode is over do not add next q-values to the target
        y_val = replay_rewards + replay_continues * self.gamma * q_best_next
        with tf.GradientTape() as tape:
            # calculate current q-values
            q_values = self.online_network(replay_state)
            # get the q-values of the executed actions
            q_values_masked = tf.reduce_sum(q_values * tf.one_hot(replay_action, self.n_actions), axis=1, keepdims=True)
            # calculate loss
            loss = self.loss_layer(q_values_masked, y_val)
        # compute gradients
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.online_network.trainable_variables))
        return loss

    def update(self, *args, **kwargs):
        return self.train_op(*args, **kwargs)

    def save(self, filepath):
        """
        saves the model to corresponding folder
        :param filepath: (string) path to the folder to save the model to
        :return: None
        """
        tf.saved_model.save(self.online_network, export_dir=filepath)

    def load(self, filepath):
        """
        load the model from corresponding path
        :param filepath: (string) path to the folder with saved model
        :return: (tf.keras.model) saved model
        """
        self.online_network = tf.saved_model.load(filepath)
        return self.online_network


class DoubleDQN(DeepQNetwork):
    def __init__(self, target_network_update_freq=8000,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_network = self.create_model('target_dqn')
        # At the begining of the training set the same weights to both networks
        self.hard_update_target_network()
        self.steps = 0
        self.target_network_update_freq = target_network_update_freq

    @tf.function
    def get_best_action_value(self, state):
        return tf.reduce_max(self.target_network(state), axis=1)

    def update(self, *args, **kwargs):
        loss = super().update(*args, **kwargs)
        if self.steps % self.target_network_update_freq == 0:
            self.hard_update_target_network()
        self.steps += 1
        return loss

    def hard_update_target_network(self):
        self.target_network.set_weights(self.online_network.get_weights())


class DuelingDDQN(DoubleDQN):
    def create_model(self, model_name):
        input_state = tf.keras.Input(shape=self.state_shape,
                                     batch_size=None,
                                     name='state_input',
                                     dtype=tf.uint8)

        x = tf.divide(tf.cast(input_state, dtype=tf.float32), tf.constant(255., dtype=tf.float32))

        x = Conv2D(16, kernel_size=(8, 8), strides=(4, 4),
                   padding='valid', activation='relu', name='conv1')(x)
        x = Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                   padding='valid', activation='relu', name='conv2')(x)
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                   padding='valid', activation='relu', name='conv3')(x)

        x = Flatten()(x)

        x = Dense(512, activation='tanh', name='hidden_dense')(x)

        value = Dense(1, activation='linear', name='hidden_dense_value')(x)
        advantage = Dense(self.n_actions, activation='linear', name='dense_advantage')(x)

        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        q_values = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))

        q_network = Model(inputs=[input_state], outputs=[q_values], name=model_name)

        return q_network


class NoisyDuelingDDQN(DuelingDDQN):

    def create_model(self, model_name):
        input_state = tf.keras.Input(shape=self.state_shape,
                                     batch_size=None,
                                     name='state_input',
                                     dtype=tf.uint8)

        x = tf.divide(tf.cast(input_state, dtype=tf.float32), tf.constant(255., dtype=tf.float32))

        x = Conv2D(16, kernel_size=(8, 8), strides=(4, 4),
                   padding='valid', activation='relu', name='conv1')(x)
        x = Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                   padding='valid', activation='relu', name='conv2')(x)
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                   padding='valid', activation='relu', name='conv3')(x)

        x = Flatten()(x)

        x = NoisyDense(512, activation='tanh', name='hidden_dense')(x)

        value = NoisyDense(1, activation='linear', name='hidden_dense_value')(x)
        advantage = NoisyDense(self.n_actions, activation='linear', name='dense_advantage')(x)

        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        q_values = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))

        q_network = Model(inputs=[input_state], outputs=[q_values], name=model_name)

        return q_network
