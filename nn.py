from typing import Optional

import tensorflow as tf


class MinMaxNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, min: float, max: float, newmin: float, newmax: float, round: bool = False, *args, **kwargs):
        super(MinMaxNormalizationLayer, self).__init__(*args, **kwargs)
        self.min = min
        self.max = max
        self.newmin = newmin
        self.newmax = newmax

        self.round = round

    def call(self, input, **kwargs):
        res = (input - self.min)/(self.max - self.min) * (self.newmax - self.newmin) + self.newmin

        if self.round:
            res = tf.math.round(res)

        return res


class SwitchLayer(tf.keras.layers.Layer):
    def __init__(self, run: Optional[tf.keras.layers.Layer] = None, train: Optional[tf.keras.layers.Layer] = None, *args, **kwargs):
        super(SwitchLayer, self).__init__(*args, **kwargs)

        self.run = run
        self.train = train

    def call(self, input, **kwargs):
        kwargs.setdefault("training", False)

        if kwargs["training"]:
            if self.train is None:
                return input
            else:
                return self.train(input)

        else:
            if self.run is None:
                return input
            else:
                return self.run(input)
