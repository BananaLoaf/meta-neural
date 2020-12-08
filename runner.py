import os
from datetime import datetime
from typing import Tuple, Optional, Type, Dict, Generator, Any

from pathlib import Path
import tensorflow as tf
import tflite_runtime.interpreter as tflite

from metaneural.config import DefaultConfig, ConverterConfig, ResumeConfig


class Runner:
    def __init__(self, config: Type[DefaultConfig], run_directory: Path):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.config = config
        # TODO something
        # if config.xla_jit:
        #     os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

        ################################################################
        # Paths
        self.run_path = run_directory
        self.run_path.mkdir(exist_ok=True, parents=True)
        self.samples_path = self.run_path.joinpath("samples")
        self.model_path = self.run_path.joinpath("model")
        self.checkpoint_path = self.run_path.joinpath("checkpoint")

        ################################################################
        self._strategy = self._init_strategy()
        self.model, self.optimizer = self.init()
        self.quantize_model()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                             directory=self.checkpoint_path,
                                                             max_to_keep=self.config.epochs,
                                                             checkpoint_name="model")

        ################################################################
        self.dataset = self.init_dataset()
        self.repr_dataset = self.init_repr_dataset()

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
    @with_strategy
    def init(self) -> Tuple[tf.keras.Model, tf.keras.optimizers.Optimizer]:
        raise NotImplementedError

    def init_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError

    def init_repr_dataset(self) -> Generator:
        return None

    def quantize_model(self):
        if bool(self.config.q_aware_train[0]):
            import tensorflow_model_optimization as tfmot
            self.model = tfmot.quantization.keras.quantize_model(self.model)

    ################################################################
    @classmethod
    def new_run(cls, config: Type[DefaultConfig]):
        self = cls(config=config,
                   run_directory=Path(f"runs/{config.name}_{datetime.now().replace(microsecond=0).isoformat()}"))
        self.summary(plot=True)

        try:
            self.train()
        except KeyboardInterrupt:
            print("\nStopping...")

    @classmethod
    def resume(cls, config: Type[DefaultConfig], resume_config: ResumeConfig):
        self = cls(config=config,
                   run_directory=Path(resume_config.path))
        self.restore(step=resume_config.checkpoint_epoch)
        self.summary()

        if resume_config.checkpoint_epoch is not None:
            config.epoch = resume_config.checkpoint_epoch

        try:
            self.train(resume=True)
        except KeyboardInterrupt:
            print("\nStopping...")

    @classmethod
    def convert(cls, config: Type[DefaultConfig], converter_config: ConverterConfig):
        self = cls(config=config,
                   run_directory=Path(converter_config.path))
        self.restore(step=converter_config.checkpoint_epoch)

        print("Saving models")
        self.save_model()

        print("Converting models")
        self.convert_model(converter_config)

    ################################################################
    def train(self, resume: bool = False):
        raise NotImplementedError

    ################################################################
    # Saving, snapping, etc
    def summary(self, plot: bool = False):
        self._summary(self.model, "model", plot)

    def _summary(self, model: tf.keras.Model, name: str, plot: bool):
        model.summary()
        if plot:
            tf.keras.utils.plot_model(model, to_file=self.run_path.joinpath(f"{name}.png"), show_shapes=True, dpi=64)

    def save_config(self, epoch: int):
        self.config.epoch = epoch
        self.config.save(self.run_path.joinpath("config.json"))

    # @merge
    def snap(self, epoch: int):
        print("\nSaving checkpoint")
        self.checkpoint_manager.save(epoch)

    @with_strategy
    def restore(self, step: Optional[int] = None):
        if step is None:
            checkpoint_path = self.checkpoint_manager.latest_checkpoint
        else:
            checkpoint_path = str(self.checkpoint_manager._directory.joinpath(f"model-{step}"))

        self.checkpoint.restore(checkpoint_path)

    def save_model(self):
        self._save_model(self.model, "model")

    def _save_model(self, model: tf.keras.Model, name: str):
        model.save(str(self.model_path.joinpath(name)), save_format="tf")

    def convert_model(self, config: ConverterConfig):
        self._convert_model(self.model, "model", self.repr_dataset,
                            dyn_range_q=config.dyn_range_q, f16_q=config.f16_q,
                            int_float_q=config.int_float_q, int_q=config.int_q)

    def _convert_model(self, model: tf.keras.Model, name: str, repr_dataset: Optional,
                       dyn_range_q: bool, f16_q: bool, int_float_q: bool, int_q: Optional[str]):
        # TFLite
        converter = tflite.TFLiteConverter.from_keras_model(model)
        with self.model_path.joinpath(f"{name}.tflite").open("wb") as file:
            file.write(converter.convert())

        # TFLite quantizised
        # Dynamic range quantization
        if dyn_range_q:
            converter = tflite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tflite.Optimize.DEFAULT]
            with self.model_path.joinpath(f"{name}.dyn-range-q.tflite").open("wb") as file:
                file.write(converter.convert())

        # Float16 quantization
        if f16_q:
            converter = tflite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tflite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            with self.model_path.joinpath(f"{name}.f16-q.tflite").open("wb") as file:
                file.write(converter.convert())

        if repr_dataset is None:
            print(f"No representative dataset for {name}")
        else:
            # Full integer quantization, integer with float fallback
            if int_float_q:
                converter = tflite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tflite.Optimize.DEFAULT]
                converter.repr_dataset = repr_dataset
                with self.model_path.joinpath(f"{name}.int-float-q.tflite").open("wb") as file:
                    file.write(converter.convert())

            # Full integer quantization, integer only
            if int_q is not None:
                converter = tflite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tflite.Optimize.DEFAULT]
                converter.repr_dataset = repr_dataset
                converter.target_spec.supported_ops = [tflite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = getattr(tf, int_q)
                converter.inference_output_type = getattr(tf, int_q)
                with self.model_path.joinpath(f"{name}.int-q.tflite").open("wb") as file:
                    file.write(converter.convert())
