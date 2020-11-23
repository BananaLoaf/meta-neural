from typing import Optional

import tensorflow as tf


class Switch(tf.keras.layers.Layer):
    def __init__(self, run: Optional[tf.keras.layers.Layer] = None, train: Optional[tf.keras.layers.Layer] = None, *args, **kwargs):
        self.run = run
        self.train = train

        if run is not None and train is not None:
            assert run.input_shape == train.input_shape, "Input tensor shape mismatch"
            assert run.output_shape == train.output_shape, "Output tensor shape mismatch"

        super(Switch, self).__init__(*args, **kwargs)

    def call(self, inputs, training=False):
        if training:
            return inputs if self.train is None else self.train(inputs)

        else:
            return inputs if self.run is None else self.run(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(Switch, self).get_config()
        config.update(run=self.run, train=self.train)
        return config
