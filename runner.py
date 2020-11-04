import os
from datetime import datetime
from typing import Tuple, Optional, Type, Dict, Generator, Any

from pathlib import Path
import tensorflow as tf
import tflite_runtime.interpreter as tflite

from metaneural.config import DefaultConfig, ConverterConfig


class RegistryEntry:
    def __init__(self,
                 name: str,
                 model: tf.keras.models.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 repr_dataset: Optional[Generator] = None):
        self.name = name
        self.model = model
        self.optimizer = optimizer

        self.q_aware_train = False
        self.repr_dataset = repr_dataset

        self.checkpoint = None
        self.checkpoint_manager = None

    def set_checkpoint(self, checkpoints_path: Path, max_to_keep: int):
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                             directory=checkpoints_path.joinpath(self.name),
                                                             max_to_keep=max_to_keep,
                                                             checkpoint_name=self.name)

    def quantize_model(self, q_aware_train: bool):
        self.q_aware_train = q_aware_train

        if q_aware_train:
            import tensorflow_model_optimization as tfmot
            self.model = tfmot.quantization.keras.quantize_model(self.model)


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
        self.samples_path.mkdir(exist_ok=True, parents=True)
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

        for i, re in enumerate(self.model_registry):
            re.set_checkpoint(self.run_path.joinpath("checkpoints"), max_to_keep=self.config.steps)
            re.quantize_model(bool(config.q_aware_train[i]))

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
        Merge args across replicas and run merge_fn in a cross-replica context.
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
    def init_dataloader(self) -> Any:
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

        try:
            self.train(resume=True)
        except KeyboardInterrupt:
            print("\nSaving and stopping...")
            self._save_models()

    @classmethod
    def convert(cls, config: Type[DefaultConfig], converter_config: ConverterConfig, run_directory: Path):
        self = cls(config=config,
                   run_directory=run_directory)
        self._load_models()
        self._convert_models(converter_config)

    ################################################################
    def train_step(self) -> dict:
        raise NotImplementedError

    def train(self, resume: bool = False):
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

    def _load_models(self):
        for re in self.model_registry:
            re.model = tf.keras.models.load_model(str(self.model_path.joinpath(re.name)))

    def _save_models(self):
        for re in self.model_registry:
            re.model.save(str(self.model_path.joinpath(re.name)), save_format="tf")

    def _convert_models(self, config: ConverterConfig):
        for re in self.model_registry:
            name = re.name + ".q-aware" if re.q_aware_train else ""

            # TFLite
            converter = tflite.TFLiteConverter.from_keras_model(re.model)
            with self.model_path.joinpath(f"{name}.tflite").open("wb") as file:
                file.write(converter.convert())

            # TFLite quantizised
            # Dynamic range quantization
            if config.dyn_range_q:
                converter = tflite.TFLiteConverter.from_keras_model(re.model)
                converter.optimizations = [tflite.Optimize.DEFAULT]
                with self.model_path.joinpath(f"{name}.dyn-range-q.tflite").open("wb") as file:
                    file.write(converter.convert())

            # Full integer quantization, integer with float fallback
            if config.int_float_q:
                converter = tflite.TFLiteConverter.from_keras_model(re.model)
                converter.optimizations = [tflite.Optimize.DEFAULT]
                converter.repr_dataset = re.repr_dataset
                with self.model_path.joinpath(f"{name}.int-float-q.tflite").open("wb") as file:
                    file.write(converter.convert())

            # Full integer quantization, integer only
            if config.int_q is not None:
                converter = tflite.TFLiteConverter.from_keras_model(re.model)
                converter.optimizations = [tflite.Optimize.DEFAULT]
                converter.repr_dataset = re.repr_dataset
                converter.target_spec.supported_ops = [tflite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = getattr(tf, config.int_q)
                converter.inference_output_type = getattr(tf, config.int_q)
                with self.model_path.joinpath(f"{name}.int-q.tflite").open("wb") as file:
                    file.write(converter.convert())

            # Float16 quantization
            if config.f16_q:
                converter = tflite.TFLiteConverter.from_keras_model(re.model)
                converter.optimizations = [tflite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                with self.model_path.joinpath(f"{name}.f16-q.tflite").open("wb") as file:
                    file.write(converter.convert())
