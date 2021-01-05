import tensorflow as tf
from dqn.NoisyDense import NoisyDense


class DQN:
    def __init__(
        self,
        input_shape,
        n_outputs,
        double_q=False,
        copy_steps=500,
        dueling_architecture=False,
        noisy=False,
        gradient_clipping=False,
        gradient_clipping_norm=10.0,
        learning_rate=0.00025,
        gamma=0.99,
        loss="huber",
        optimizer="adam",
        mlp_n_hidden=(32, 32, 64),
        mlp_act_f="relu",
        cnn_number_of_maps=(32, 64, 64),
        cnn_kernel_size=(8, 4, 3),
        cnn_kernel_stride=(4, 2, 1),
        cnn_padding="VALID",
        cnn_activation="elu",
        mlp_value_n_hidden=512,
        mlp_value_act_f="tanh",
        use_cnn=True,
    ):

        self.gamma = gamma

        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.get(
            {"class_name": optimizer, "config": {"learning_rate": self.lr}}
        )

        if loss == "huber":
            self.loss_layer = tf.keras.losses.Huber()
        elif loss == "mse":
            self.loss_layer = tf.keras.losses.MeanSquaredError()

        # shape of the input without batch_size
        self.input_shape = tuple(input_shape)

        # MLP parameters
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_act_f = mlp_act_f

        self.mlp_value_n_hidden = mlp_value_n_hidden
        self.mlp_value_act_f = mlp_value_act_f
        self.n_outputs = n_outputs

        # CNN parameters
        self.cnn_number_of_maps = cnn_number_of_maps
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_kernel_stride = cnn_kernel_stride
        self.cnn_padding = cnn_padding
        self.cnn_activation = cnn_activation

        self.is_gradient_clipping = gradient_clipping
        self.gradient_clipping_norm = gradient_clipping_norm

        self.is_double_q = double_q
        self.copy_steps = copy_steps

        self.is_dueling = dueling_architecture

        if noisy:
            self.dense = NoisyDense
        else:
            self.dense = tf.keras.layers.Dense

        self.online_network = self.create_model(
            "dqn/online", dueling=self.is_dueling, save_layers=True, use_cnn=use_cnn
        )
        if self.is_double_q:
            self.target_network = self.create_model(
                "dqn/target",
                dueling=self.is_dueling,
                save_layers=False,
                use_cnn=use_cnn,
            )
            self.copy_to_target()
        else:
            self.target_network = self.online_network

        self.loss_history = []

    def update(self):
        """
        Perform necessary network updates per each step
        :return: None
        """
        if self.is_double_q:
            # Hard copy all the weights from online to target
            self.copy_to_target()

    def copy_to_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    @tf.function
    def get_action_online(self, state):
        return tf.argmax(self.online_network(state), axis=1)

    @tf.function
    def get_action_target(self, state):
        return tf.argmax(self.target_network(state), axis=1)

    @tf.function
    def calculate_online(self, state):
        return self.online_network(state)

    @tf.function
    def calculate_target(self, state):
        return self.target_network(state)

    @tf.function
    def train_op(
        self,
        replay_state,
        replay_action,
        replay_rewards,
        replay_next_state,
        terminal,
        indices,
        loss_weights=None,
    ):
        # We assume that if have received a uint8 this means we need to
        if replay_state.dtype == "uint8":
            replay_state = replay_state.astype("float32") / 255.0

        replay_action = tf.reshape(replay_action, (-1,))
        replay_continues = tf.reshape(
            1.0 - tf.cast(terminal, dtype=tf.float32), (-1, 1)
        )
        replay_rewards = tf.reshape(replay_rewards, (-1, 1))
        next_q_values = self.target_network(replay_next_state)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1, keepdims=True)
        y_val = replay_rewards + replay_continues * self.gamma * max_next_q_values

        with tf.GradientTape() as tape:
            predicted = self.online_network(replay_state)
            predicted = tf.reduce_sum(
                predicted * tf.one_hot(replay_action, self.n_outputs),
                axis=1,
                keepdims=True,
            )
            td_error = tf.abs(predicted - y_val)

            loss = self.loss_layer(predicted, y_val, sample_weight=loss_weights)
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        if self.is_gradient_clipping:
            gradients, global_norm = tf.clip_by_global_norm(
                gradients, self.gradient_clipping_norm
            )
        self.optimizer.apply_gradients(
            zip(gradients, self.online_network.trainable_variables)
        )
        # Train the online DQN
        return loss, td_error, indices

    def train_online(self, batch):
        loss, td_error, indices = self.train_op(*batch)
        self.loss_history.append(loss)
        return loss, td_error.numpy(), indices.numpy()

    def create_model(
        self,
        model_name,
        dueling=False,
        save_layers=True,
        use_cnn=True,
        print_summary=False,
    ):

        state_input = tf.keras.Input(
            shape=self.input_shape,
            batch_size=None,
            name="state_input",
            dtype=tf.float32,
        )

        x = state_input
        if use_cnn:
            x, feature_layers = self.create_conv_stacks(x, flatten_output=True)
        else:
            x, feature_layers = self.create_dense_stacks(x)

        if dueling:
            q_values_output, dense_layers = self.create_value_dense_dueling(x)
        else:
            q_values_output, dense_layers = self.create_value_dense(x)

        layers = feature_layers + dense_layers

        if save_layers:
            self.layers = layers

        q_network = tf.keras.models.Model(
            inputs=[state_input], outputs=[q_values_output], name=model_name
        )

        if print_summary:
            q_network.summary()

        return q_network

    def create_value_dense(self, flattened_input):
        layers = []

        dense_hidden_layer = self.dense(
            self.mlp_value_n_hidden,
            activation=self.mlp_value_act_f,
            use_bias=True,
            bias_initializer="zeros",
            name="dense_hidden_layer",
        )
        layers.append(dense_hidden_layer)

        hidden_state = dense_hidden_layer(flattened_input)
        q_values_layer = self.dense(
            self.n_outputs,
            activation=None,
            use_bias=True,
            bias_initializer="zeros",
            name="q_values_layer",
        )
        layers.append(q_values_layer)

        q_values = q_values_layer(hidden_state)
        return q_values, layers

    def create_value_dense_dueling(self, flattened_input):
        layers = []

        # Hidden  for value estimation
        value_hidden = self.dense(
            self.mlp_value_n_hidden,
            activation=self.mlp_value_act_f,
            bias_initializer="zeros",
            name="value_hidden",
        )
        layers.append(value_hidden)
        value_stream = value_hidden(flattened_input)

        # Hidden for advantage estimation
        advantage_hidden = self.dense(
            self.mlp_value_n_hidden,
            activation=self.mlp_value_act_f,
            bias_initializer="zeros",
            name="advantage_hidden",
        )
        layers.append(advantage_hidden)
        advantage_stream = advantage_hidden(flattened_input)

        # A(s,a)
        advantage_layer = self.dense(
            self.n_outputs,
            bias_initializer="zeros",
            name="advantage",
        )
        layers.append(advantage_layer)
        advantage = advantage_layer(advantage_stream)  # linear activation

        # V(s)
        value_layer = self.dense(
            1,
            bias_initializer="zeros",
            name="value",
        )
        layers.append(value_layer)
        value = value_layer(value_stream)  # linear activation

        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        q_values = value + tf.subtract(
            advantage, tf.reduce_mean(advantage, axis=1, keepdims=True)
        )
        return q_values, layers

    def create_dense_stacks(self, state):
        layers = []
        current_output = state
        for (size,) in zip(self.mlp_n_hidden):
            dense = tf.keras.layers.Dense(
                size,
                activation=self.mlp_act_f,
                use_bias=True,
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
            )
            layers.append(dense)
            current_output = dense(current_output)
        return current_output, layers

    def create_conv_stacks(self, state, flatten_output=True):
        layers = []
        current_output = tf.cast(state, dtype=tf.float32)

        for kernel_size, kernel_stride, filters in zip(
            self.cnn_kernel_size, self.cnn_kernel_stride, self.cnn_number_of_maps
        ):
            conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size,
                strides=kernel_stride,
                padding=self.cnn_padding,
                activation=self.cnn_activation,
            )
            current_output = conv(current_output)
            layers.append(conv)

        if flatten_output:
            current_output = tf.keras.layers.Flatten()(current_output)

        return current_output, layers
