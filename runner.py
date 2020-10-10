import os
from datetime import datetime
from typing import Tuple, Optional, Type, Dict, Generator

from pathlib import Path
import tensorflow as tf

from metaneural.config import DefaultConfig
from metaneural.dataloader import DefaultDataloader


MODEL = "MODEL"
OPTIMIZER = "OPTIMIZER"
CHECKPOINT = "CHECKPOINT"
CHECKPOINT_MANAGER = "CHECKPOINT_MANAGER"
QUANTIZATION_TRAINING = "QUANTIZATION_TRAINING"


class RegistryEntry:
    def __init__(self,
                 name: str,
                 q_aware_training: bool,
                 model: tf.keras.models.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 checkpoint: tf.train.Checkpoint,
                 checkpoint_manager: tf.train.CheckpointManager,
                 representative_dataset: Optional[Generator] = None):
        self.name = name
        self.q_aware_training = q_aware_training
        self.model = model
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.checkpoint_manager = checkpoint_manager
        self.representative_dataset = representative_dataset


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
        # run_path gets created with writers
        self.train_writer = tf.summary.create_file_writer(str(self.run_path.joinpath("train")))
        self.train_writer.set_as_default()

        self.validate_writer = tf.summary.create_file_writer(str(self.run_path.joinpath("validate")))

        ################################################################
        self.dataloader = self.init_dataloader()
        self._strategy = self._init_strategy()
        self.model_registry = self.init_networks()

        if config.quant_aware_train:
            import tensorflow_model_optimization as tfmot
            for re in self.model_registry:
                if re.q_aware_training:
                    re.model = tfmot.quantization.keras.quantize_model(re.model)

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
    def init_dataloader(self) -> Type[DefaultDataloader]:
        raise NotImplementedError

    @with_strategy
    def init_networks(self) -> Tuple[RegistryEntry, ...]:
        raise NotImplementedError

    ################################################################
    @classmethod
    def new_run(cls, config: Type[DefaultConfig]):
        self = cls(config=config,
                   run_directory=Path(f"runs/{config.name}_{datetime.now().replace(microsecond=0).isoformat()}"))
        self._summary(plot=True)

        try:
            self.train()
        except KeyboardInterrupt:
            print("\nSaving and stopping...")
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

    def train(self):
        raise NotImplementedError

    def validate(self) -> dict:
        raise NotImplementedError

    ################################################################
    def sample(self, step: int):
        raise NotImplementedError

    ################################################################
    # Saving, snapping, etc
    def _summary(self, plot: bool = False):
        for re in self.model_registry:
            re.model.summary()
            if plot:
                tf.keras.utils.plot_model(re.model, to_file=self.run_path.joinpath(re.name), show_shapes=True, dpi=64)

    def _save_config(self):
        self.config.save(self.run_path.joinpath("config.json"))

    @merge
    def _snap(self, step: int):
        for re in self.model_registry:
            re.checkpoint_manager.save(step)

    @with_strategy
    def _restore(self):
        for re in self.model_registry:
            re.checkpoint.restore(re.checkpoint_manager.latest_checkpoint)

    def _save_models(self):
        for re in self.model_registry:
            # Tensorflow
            re.model.save(str(self.model_path.joinpath(re.name)), save_format="tf")

            if self.config.save_tflite or self.config.save_tflite_q:
                # TFLite
                if self.config.save_tflite:
                    converter = tf.lite.TFLiteConverter.from_keras_model(re.model)
                    with self.model_path.joinpath(f"{re.name}.tflite").open("wb") as file:
                        file.write(converter.convert())

                # TFLite quantizised
                if self.config.save_tflite_q:
                    converter = tf.lite.TFLiteConverter.from_keras_model(re.model)

                    # Dynamic range quantization
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    with self.model_path.joinpath(f"{re.name}.q.tflite").open("wb") as file:
                        file.write(converter.convert())

                    converter.representative_dataset = re.representative_dataset
                    # Full integer quantization, integer with float fallback
                    if self.config.int_float_q:
                        with self.model_path.joinpath(f"{re.name}.int-float-q.tflite").open("wb") as file:
                            file.write(converter.convert())

                    # Full integer quantization, integer only
                    if self.config.int_q:
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                        converter.inference_input_type = tf.int8  # or tf.uint8
                        converter.inference_output_type = tf.int8  # or tf.uint8
                        with self.model_path.joinpath(f"{re.name}.int-q.tflite").open("wb") as file:
                            file.write(converter.convert())

                    # Float16 quantization
                    if self.config.f16_q:
                        converter = tf.lite.TFLiteConverter.from_keras_model(re.model)
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        converter.target_spec.supported_types = [tf.float16]
                        with self.model_path.joinpath(f"{re.name}.f16-q.tflite").open("wb") as file:
                            file.write(converter.convert())
