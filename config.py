from typing import Tuple, Any, Callable
from argparse import ArgumentParser
import json

from pathlib import Path


ARGS = "ARGS"
KWARGS = "KWARGS"
GROUP_NAME = "GROUP_NAME"
# EXCLUSIVE_GROUP = "EXCLUSIVE_GROUP"
CONSTANT = "CONSTANT"
SAVE = "SAVE"

TYPE = "type"
ACTION = "action"
REQUIRED = "required"
DEFAULT = "default"
CHOICES = "choices"
HELP = "help"


class ConfigBuilder:
    def __init__(self):
        for field, scheme, value in self.get_field_scheme_value():
            setattr(self, field, None)
            setattr(self.__class__, field, self.set_defaults(scheme))

    def get_attrs(self) -> str:
        for attr, value in {**vars(self.__class__), **vars(self.__class__.__base__)}.items():
            if not (attr.startswith("__") and attr.endswith("__")) and not isinstance(value, Callable):
                yield attr

    @staticmethod
    def set_defaults(scheme: dict) -> dict:
        """Set all default values for a scheme"""
        scheme.setdefault(SAVE, True)
        return scheme

    def get_field_scheme_value(self) -> Tuple[str, dict, Any]:
        """Fetches all available fields, schemes and values"""
        for attr in self.get_attrs():
            yield attr, getattr(self.__class__, attr), getattr(self, attr)

    @classmethod
    def cli(cls, description: str = ""):
        self = cls()

        ################################################################
        # Create parser
        parser = ArgumentParser(description=description)
        groups = {}

        for field, scheme, value in self.get_field_scheme_value():
            # Set constants and skip
            if CONSTANT in scheme.keys():
                setattr(self, field, scheme[CONSTANT])
                continue

            # Create group and set as target for new argument
            if GROUP_NAME in scheme.keys():
                groups.setdefault(scheme[GROUP_NAME], parser.add_argument_group(scheme[GROUP_NAME]))
                target_parser = groups[scheme[GROUP_NAME]]

            else:
                target_parser = parser

            target_parser.add_argument(*scheme[ARGS], **scheme[KWARGS], dest=field)

        ################################################################
        # Parse
        args = parser.parse_args()
        for field, value in vars(args).items():
            setattr(self, field, value)

        return self

    @classmethod
    def load(cls, path: Path):
        self = cls()
        self._cleanup()

        with path.open("r") as file:
            data = json.load(file)

        for field, scheme, value in self.get_field_scheme_value():
            try:
                setattr(self, field, data[field])
            except KeyError:
                raise KeyError(f"Config is missing required key '{field}'")

        return self

    def save(self, path: Path):
        self._cleanup()
        with path.open("w") as file:
            json.dump(self.to_json(), file, indent=4)

    def to_json(self) -> dict:
        data = {}
        for field, scheme, value in self.get_field_scheme_value():
            data[field] = getattr(self, field)
        return data

    def __repr__(self):
        return self.to_json()

    def _cleanup(self):
        to_remove = []
        for field, scheme, value in self.get_field_scheme_value():
            if not scheme[SAVE]:
                to_remove.append(field)

        for field in to_remove:
            delattr(self, field)
            delattr(self.__class__, field)


class DefaultConfig(ConfigBuilder):
    """
    This config implementation allows to easily trace param usage with the help of IDE

    Examples:
    name = {GROUP_NAME: "Model",                                       # Not required
            ARGS: ["--name"],                                          # Required
            KWARGS: {TYPE: str, REQUIRED: True, HELP: "Model name"}},  # Required
            SAVE: False                                                # Saving in config.json, gets wiped on save (default: True)

    # Does not provide cli param, just exists
    step = {CONSTANT: 0,
            SAVE; False}

    # Not used
    device = {GROUP_NAME: "Device params",
              EXCLUSIVE_GROUP: [
                  {ARGS: ["--cpu"],
                   KWARGS: {TYPE: str, DEFAULT: "/device:CPU:0", CHOICES: [dev.name for dev in tf.config.list_logical_devices("CPU")], HELP: "CPU (default: %(default)s)"}},
                  {ARGS: ["--gpu"],
                   KWARGS: {TYPE: str, HELP: "GPUs"}}
              ],
              REQUIRED: False}  # Only used with EXCLUSIVE_GROUP, if not required, one of elements in a group must have DEFAULT value (default: True)
    """

    name = {ARGS: ["--name"],
            KWARGS: {TYPE: str, REQUIRED: True, HELP: "Model name"}}

    # Device params
    use_tpu = {GROUP_NAME: "Device params",
               ARGS: ["--use-tpu"],
               KWARGS: {ACTION: "store_true",
                        HELP: "Use Google Cloud TPU, if True, --gpu param is ignored (default: %(default)s)"}}
    tpu_name = {GROUP_NAME: "Device params",
                ARGS: ["--tpu-name"],
                KWARGS: {TYPE: str, DEFAULT: None,
                         HELP: "Google Cloud TPU name, if None and flag --use-tpu is set, will try to detect automatically (default: %(default)s)"}}
    devices = {GROUP_NAME: "Device params",
               ARGS: ["--gpu"],
               KWARGS: {TYPE: str, DEFAULT: None,
                        HELP: "Available GPUs: {}, list devices with , as delimiter"}}  # Format however is needed
    xla_jit = {GROUP_NAME: "Device params",
               ARGS: ["--xla-jit"],
               KWARGS: {ACTION: "store_true",
                        HELP: "XLA Just In Time compilation, https://www.tensorflow.org/xla (default: %(default)s)"}}

    # Training params
    step = {CONSTANT: 0}
    steps = {GROUP_NAME: "Training params",
             ARGS: ["-s", "--steps"],
             KWARGS: {TYPE: int, DEFAULT: 1_000_000, HELP: "Steps (default: %(default)s)"}}
    quantization_training = {GROUP_NAME: "Training params",
                             ARGS: ["-qt", "--quantizised-training"],
                             KWARGS: {ACTION: "store_true",
                                      HELP: "Quantization aware training, https://www.tensorflow.org/model_optimization/guide/quantization/training (default: %(default)s)"}}
    batch_size = {GROUP_NAME: "Training params",
                  ARGS: ["-b", "--batch-size"],
                  KWARGS: {TYPE: int, DEFAULT: 2, HELP: "Batch size (default: %(default)s)"}}
    checkpoint_freq = {GROUP_NAME: "Training params",
                       ARGS: ["-cf", "--checkpoint-freq"],
                       KWARGS: {TYPE: int, DEFAULT: 10_000,
                                HELP: "Checkpoint frequency in steps (default: %(default)s)"}}

    # Saving params
    save_tflite = {GROUP_NAME: "Saving params",
                   ARGS: ["--tflite"],
                   KWARGS: {ACTION: "store_true", DEFAULT: False, HELP: "Save as tflite model"}}
    save_tflite_q = {GROUP_NAME: "Saving params",
                     ARGS: ["--tflite-q"],
                     KWARGS: {ACTION: "store_true", DEFAULT: False, HELP: "Save as quantizised tflite model"}}