import tensorflow as tf
from tensorflow.keras import layers, models

def build_critic_network(obs_shape, num_dim=(64, 64), act_f="tanh"):
    state_input = layers.Input(shape=obs_shape, dtype=tf.float32)
    x = state_input
    for i, dim in enumerate(num_dim):
        x = layers.Dense(dim, activation=act_f, name=f"hidden_{i}")(x)

    out_value = layers.Dense(1, name="output")(x)

    model = models.Model(inputs=state_input, outputs=out_value, name="critic")
    model.summary()
    return model


def build_actor_network(
    obs_shape, n_outputs, num_dim=(64, 64), act_f="tanh", output_act_f="softmax"
):
    state_input = layers.Input(shape=obs_shape, dtype=tf.float32)

    x = state_input
    for i, dim in enumerate(num_dim):
        x = layers.Dense(
            dim,
            activation=act_f,
            name=f"hidden_{i}",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        )(x)

    out_actions = layers.Dense(n_outputs, activation=output_act_f, name="output")(x)

    model = models.Model(inputs=state_input, outputs=out_actions, name="actor")
    model.summary()
    return model