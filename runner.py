import os
from datetime import datetime
from typing import Tuple, Optional, Type

from pathlib import Path
import tensorflow as tf

from ..config import DefaultConfig
from ..dataloader import DefaultDataloader


MODEL = "MODEL"
OPTIMIZER = "OPTIMIZER"
CHECKPOINT = "CHECKPOINT"
CHECKPOINT_MANAGER = "CHECKPOINT_MANAGER"


class Runner:
    def __init__(self, config: Type[DefaultConfig], run_directory: Path):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.config = config
        if config.xla_jit:
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

        ################################################################
        # Paths
        self.run_path = run_directory
        self.samples_path = self.run_path.joinpath("samples")
        self.checkpoints_path = self.run_path.joinpath("checkpoints")
        self.model_path = self.run_path.joinpath("model")

        ################################################################
        self.dataloader = self.init_dataloader()
        self._strategy = self._init_strategy()
        self._model_registry = self.init_networks()

    ################################################################
    # https://www.tensorflow.org/api_docs/python/tf/distribute
    def _init_strategy(self) -> tf.distribute.Strategy:
        if self.config.use_tpu:
            kwargs = {} if self.config.tpu_name is None else {"tpu": self.config.tpu_name}
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(**kwargs)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.experimental.TPUStrategy(resolver)

        else:
            all_devices = [dev.name for dev in tf.config.list_logical_devices()]

            if self.config.devices is not None:
                devices = [f"/device:{'XLA_' if self.config.xla_jit else ''}GPU:{dev}" for dev in
                           self.config.devices.split(",")]
                for device in devices:
                    assert device in all_devices, f"Invalid device {device}"

                return tf.distribute.MirroredStrategy(devices=devices)
            else:
                device = f"/device:{'XLA_' if self.config.xla_jit else ''}CPU:0"
                assert device in all_devices, f"Invalid device {device}"

                return tf.distribute.OneDeviceStrategy(device=device)

    def with_strategy(func):
        """Run function in a strategy context"""

        def wrapper(self, *args, **kwargs):
            return self._strategy.experimental_run_v2(lambda: func(self, *args, **kwargs))

        return wrapper

    def merge(func):
        """
        Merge args across replicas and run merge_fn in a cross-replica context. Whatever that means.
        https://www.tensorflow.org/api_docs/python/tf/distribute/ReplicaContext
        """

        def wrapper(*args, **kwargs):
            def descoper(strategy: Optional[tf.distribute.Strategy] = None, *args2, **kwargs2):
                if strategy is None:
                    return func(*args2, **kwargs2)
                else:
                    with strategy.scope():
                        return func(*args2, **kwargs2)

            return tf.distribute.get_replica_context().merge_call(descoper, args, kwargs)

        return wrapper

    ################################################################
    @property
    def model_registry(self) -> Tuple[str, tf.keras.models.Model, tf.keras.optimizers.Optimizer, tf.train.Checkpoint, tf.train.CheckpointManager]:
        for model_name, reg in self._model_registry.items():
            yield model_name, reg[MODEL], reg[OPTIMIZER], reg[CHECKPOINT], reg[CHECKPOINT_MANAGER]

    def init_dataloader(self) -> Type[DefaultDataloader]:
        raise NotImplementedError

    @with_strategy
    def init_networks(self) -> dict:
        raise NotImplementedError

    ################################################################
    @classmethod
    def train(cls, config: Type[DefaultConfig]):
        self = cls(config=config,
                   run_directory=Path(f"runs/{config.name}_{datetime.now().replace(microsecond=0).isoformat()}"))
        self._summary(plot=True)

        try:
            self._train()
        except KeyboardInterrupt:
            print("Saving and stopping...")
            self._save_models()

    @classmethod
    def resume(cls, config: Type[DefaultConfig], run_directory: Path):
        self = cls(config=config,
                   run_directory=run_directory)
        self._restore()
        self._summary()
        return self

    ################################################################
    def train_step(self) -> dict:
        raise NotImplementedError

    def _train(self):
        raise NotImplementedError

    ################################################################
    # Saving, snapping, etc
    def _summary(self, plot: bool = False):
        for model_name, model, optimizer, checkpoint, checkpoint_manager in self.model_registry:
            model.summary()
            if plot:
                img_path = self.run_path.joinpath(model_name)
                tf.keras.utils.plot_model(model, to_file=str(img_path), show_shapes=True, dpi=64)

    @merge
    def _snap(self, step: int):
        for model_name, model, optimizer, checkpoint, checkpoint_manager in self.model_registry:
            checkpoint_manager.save(step)

    @with_strategy
    def _restore(self):
        for model_name, model, optimizer, checkpoint, checkpoint_manager in self.model_registry:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)

    def _save_models(self):
        for model_name, model, optimizer, checkpoint, checkpoint_manager in self.model_registry:
            # Tensorflow
            model.save(str(self.model_path.joinpath(model_name)), save_format="tf")

            # TFLite
            if self.config.save_tflite:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                with self.model_path.joinpath(f"{model_name}.tflite").open("wb") as file:
                    file.write(converter.convert())

            # TFLite quantizised
            if self.config.save_tflite_q:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                with self.model_path.joinpath(f"{model_name}_q.tflite").open("wb") as file:
                    file.write(converter.convert())

    def _save_config(self):
        self.config.save(self.run_path.joinpath("config.json"))

