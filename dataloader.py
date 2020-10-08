from typing import Tuple, Union

import tensorflow as tf


class DefaultDataloader:
    @property
    def train_split_size(self) -> int:
        raise NotImplementedError

    @property
    def validation_split_size(self) -> int:
        raise NotImplementedError

    def next(self, batch_size: int, shuffle: bool = True, validate: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, ...]]:
        raise NotImplementedError
