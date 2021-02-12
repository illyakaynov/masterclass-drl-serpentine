import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from QLearning.dqn import NoisyDense


class DeepQNetwork:
    def __init__(
        self,
        state_shape,
        n_actions,
        learning_rate=1e-4,
        gamma=0.99,
        use_cnn=True,
        mlp_n_hidden=(32, 32, 64),
        mlp_act_f="relu",
        cnn_number_of_maps=(32, 64, 64),
        cnn_kernel_size=(8, 4, 3),
        cnn_kernel_stride=(4, 2, 1),
        mlp_value_n_hidden=(256, 512),
        mlp_value_act_f="tanh",
    ):
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.use_cnn = use_cnn

        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_act_f = mlp_act_f
        self.cnn_number_of_maps = cnn_number_of_maps
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_kernel_stride = cnn_kernel_stride
        self.mlp_value_n_hidden = mlp_value_n_hidden
        self.mlp_value_act_f = mlp_value_act_f

        self.online_network = self.create_model("dqn")

        self.state_shape = state_shape

        self.gamma = tf.constant(gamma)

        self.loss_layer = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def create_model(self, model_name):
        input_obs = tf.keras.Input(
            shape=self.state_shape, batch_size=None, name="state_input", dtype=tf.float32
        )

        x = self.create_input_embedder(input_obs)

        for i, n_dim in enumerate(self.mlp_value_n_hidden):
            x = Dense(n_dim, activation=self.mlp_value_act_f, name=f"dense_value_{i}")(
                x
            )

        q_values = Dense(self.n_actions, activation="linear", name="value_output")(x)

        q_network = Model(inputs=[input_obs], outputs=[q_values], name=model_name)
        return q_network

    def create_input_embedder(self, input_tensor):
        x = input_tensor
        if self.use_cnn:
            for i, (n_filters, kernel_size, strides) in enumerate(
                zip(
                    self.cnn_number_of_maps,
                    self.cnn_kernel_size,
                    self.cnn_kernel_stride,
                )
            ):

                x = Conv2D(
                    n_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="valid",
                    activation="relu",
                    name=f"conv_{i}",
                )(input_tensor)
            x = Flatten()(x)

        else:
            for i, n_dim in enumerate(self.mlp_n_hidden):
                x = Dense(
                    n_dim,
                    activation=self.mlp_act_f,
                    name=f"dense_hidden_{i}",
                )(x)
        return x

    @tf.function
    def get_best_action(self, state):
        return tf.argmax(self.online_network(state), axis=1)

    @tf.function
    def get_best_action_value(self, state):
        return tf.reduce_max(self.online_network(state), axis=1)

    @tf.function
    def train_op(
        self, replay_state, replay_action, replay_rewards, replay_next_state, terminal
    ):
        # We assume that if have received a uint8 this means we need to normalize
        if replay_state.dtype == "uint8":
            replay_state = tf.cast(replay_state, tf.float32) / 255.0

        replay_continues = 1.0 - terminal
        # get max q-values for the next state
        q_best_next = self.get_best_action_value(replay_next_state)
        # calculate target, if episode is over do not add next q-values to the target
        y_val = replay_rewards + replay_continues * self.gamma * q_best_next
        with tf.GradientTape() as tape:
            # calculate current q-values
            q_values = self.online_network(replay_state)
            # get the q-values of the executed actions
            q_values_masked = tf.reduce_sum(
                q_values * tf.one_hot(replay_action, self.n_actions),
                axis=1,
                keepdims=True,
            )
            # calculate loss
            loss = self.loss_layer(q_values_masked, y_val)
        # compute gradients
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        # apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.online_network.trainable_variables)
        )
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
    def __init__(self, target_network_update_freq=8000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_network = self.create_model("target_dqn")
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
        input_obs = tf.keras.Input(
            shape=self.state_shape, batch_size=None, name="state_input", dtype=tf.float32
        )

        x = self.create_input_embedder(input_obs)

        x = Flatten()(x)

        for i, n_dim in enumerate(self.mlp_value_n_hidden):
            x = Dense(n_dim, activation=self.mlp_value_act_f, name=f"dense_value_{i}")(
                x
            )

        value = Dense(1, activation="linear", name="hidden_dense_value")(x)
        advantage = Dense(self.n_actions, activation="linear", name="dense_advantage")(
            x
        )

        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        q_values = value + tf.subtract(
            advantage, tf.reduce_mean(advantage, axis=1, keepdims=True)
        )

        q_network = Model(inputs=[input_obs], outputs=[q_values], name=model_name)

        return q_network


class NoisyDuelingDDQN(DuelingDDQN):
    def create_model(self, model_name):
        input_obs = tf.keras.Input(
            shape=self.state_shape, batch_size=None, name="state_input", dtype=tf.uint8
        )

        x = self.create_input_embedder(input_obs)

        x = Flatten()(x)

        for i, n_dim in enumerate(self.mlp_value_n_hidden):
            x = NoisyDense(n_dim, activation=self.mlp_value_act_f, name=f"dense_value_{i}")(
                x
            )

        value = NoisyDense(1, activation="linear", name="hidden_dense_value")(x)
        advantage = NoisyDense(
            self.n_actions, activation="linear", name="dense_advantage"
        )(x)

        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        q_values = value + tf.subtract(
            advantage, tf.reduce_mean(advantage, axis=1, keepdims=True)
        )

        q_network = Model(inputs=[input_obs], outputs=[q_values], name=model_name)

        return q_network
