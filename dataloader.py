from typing import Tuple, Union

import tensorflow as tf


class DefaultDataloader:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __len__(self):
        raise NotImplementedError

    def __next__(self) -> Union[tf.Tensor, Tuple[tf.Tensor, ...]]:
        raise NotImplementedError

    def with_batch_size(dl, n: int):
        class Context:
            old_batch_size = dl.batch_size
            new_batch_size = n

            def __enter__(self):
                dl.batch_size = self.new_batch_size

            def __exit__(self, exc_type, exc_val, exc_tb):
                dl.batch_size = self.old_batch_size

        return Context()
