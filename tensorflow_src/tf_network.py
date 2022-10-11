import tensorflow as tf


class NetModel(tf.keras.Model):
    def __init__(self, feature_size):
        super(NetModel, self).__init__()

        self.feature_size = feature_size

        model = []
        model += [
            tf.keras.layers.Conv2D(filters=self.feature_size, kernel_size=3, strides=2, padding='SAME', use_bias=True),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=self.feature_size * 2, kernel_size=3, strides=2, padding='SAME',
                                   use_bias=True),
            tf.keras.layers.Activation(tf.keras.activations.relu)
            ]

        model += [tf.keras.layers.Flatten(),
                  tf.keras.layers.Dense(units=10, use_bias=True)]
        model = tf.keras.Sequential(model)

        self.model = tf.keras.Sequential(model)

    def call(self, x):
        x = self.model(x)
        return x