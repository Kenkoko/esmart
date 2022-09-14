from __future__ import annotations

import collections
import copy
import datetime
import os
import time
import uuid
import sys
import re
from enum import Enum

import numpy as np

import yaml
from typing import Any, List, Dict, Optional, Union

class Config:
    
    def __init__(self, folder: Optional[str] = None, load_default=True) -> None:
        """Initialize with the default configuration"""
        import esmart
        from esmart.misc import filename_in_module
        if load_default:
            with open(filename_in_module(esmart, "config-default.yaml"), "r") as file:
                self.options: Dict[str, Any] = yaml.load(file, Loader=yaml.SafeLoader)
        else:
            self.options = {}
        self.folder = folder

        self.log_folder: Optional[
            str
        ] = None  # None means use self.folder; used for kge.log, trace.yaml

        self.log_prefix: str = None

    def get(self, key: str, remove_plusplusplus=True) -> Any:
        """Obtain value of specified key.
        Nested dictionary values can be accessed via "." (e.g., "job.type"). Strips all
        '+++' keys unless `remove_plusplusplus` is set to `False`.
        """
        result = self.options
        for name in key.split("."):
            try:
                result = result[name]
            except KeyError:
                raise KeyError(f"Error accessing {name} for key {key}")

        if remove_plusplusplus and isinstance(result, collections.abc.Mapping):

            def do_remove_plusplusplus(option):
                if isinstance(option, collections.abc.Mapping):
                    option.pop("+++", None)
                    for values in option.values():
                        do_remove_plusplusplus(values)

            result = copy.deepcopy(result)
            do_remove_plusplusplus(result)

        return result

    def get_default(self, key: str) -> Any:
        """Returns the value of the key if present or default if not.
        The default value is looked up as follows. If the key has form ``parent.field``,
        see if there is a ``parent.type`` property. If so, try to look up ``field``
        under the key specified there (proceeds recursively). If not, go up until a
        `type` field is found, and then continue from there.
        """
        try:
            return self.get(key)
        except KeyError as e:
            last_dot_index = key.rfind(".")
            if last_dot_index < 0:
                raise e
            parent = key[:last_dot_index]
            field = key[last_dot_index + 1 :]
            while True:
                # self.log("Looking up {}/{}".format(parent, field))
                try:
                    parent_type = self.get(parent + "." + "type")
                    # found a type -> go to this type and lookup there
                    new_key = parent_type + "." + field
                    last_dot_index = new_key.rfind(".")
                    parent = new_key[:last_dot_index]
                    field = new_key[last_dot_index + 1 :]
                except KeyError:
                    # no type found -> go up hierarchy
                    last_dot_index = parent.rfind(".")
                    if last_dot_index < 0:
                        raise e
                    field = parent[last_dot_index + 1 :] + "." + field
                    parent = parent[:last_dot_index]
                    continue
                try:
                    value = self.get(parent + "." + field)
                    # uncomment this to see where defaults are taken from
                    # self.log(
                    #     "Using value of {}={} for key {}".format(
                    #         parent + "." + field, value, key
                    #     )
                    # )
                    return value
                except KeyError:
                    # try further
                    continue

    def exists(self, key: str, remove_plusplusplus=True) -> bool:
        try:
            self.get(key, remove_plusplusplus)
            return True
        except KeyError:
            return False

    Overwrite = Enum("Overwrite", "Yes No Error")

    def set(
        self, key: str, value, create=False, overwrite=Overwrite.Yes, log=False
    ) -> Any:

        """Set value of specified key.
        Nested dictionary values can be accessed via "." (e.g., "job.type").
        If ``create`` is ``False`` , raises :class:`ValueError` when the key
        does not exist already; otherwise, the new key-value pair is inserted
        into the configuration.
        """
        from esmart.misc import is_number

        splits = key.split(".")
        data = self.options

        # flatten path and see if it is valid to be set
        path = []
        for i in range(len(splits) - 1):
            if splits[i] in data:
                create = create or "+++" in data[splits[i]]
            else:
                if create:
                    data[splits[i]] = dict()
                else:
                    msg = (
                        "Key '{}' cannot be set because key '{}' does not exist "
                        "and no new keys are allowed to be created "
                    ).format(key, ".".join(splits[: (i + 1)]))
                    if i == 0:
                        raise KeyError(msg + "at root level.")
                    else:
                        raise KeyError(
                            msg + "under key '{}'.".format(".".join(splits[:i]))
                        )

            path.append(splits[i])
            data = data[splits[i]]

        # check correctness of value
        try:
            current_value = data.get(splits[-1])
        except:
            raise Exception(
                "These config entries {} {} caused an error.".format(data, splits[-1])
            )

        if current_value is None:
            if not create:
                msg = (
                    f"Key '{key}' cannot be set because it does not exist and "
                    "no new keys are allowed to be created "
                )
                if len(path) == 0:
                    raise KeyError(msg + "at root level.")
                else:
                    raise KeyError(msg + ("under key '{}'.").format(".".join(path)))

            if isinstance(value, str) and is_number(value, int):
                value = int(value)
            elif isinstance(value, str) and is_number(value, float):
                value = float(value)
        else:
            if (
                isinstance(value, str)
                and isinstance(current_value, float)
                and is_number(value, float)
            ):
                value = float(value)
            elif (
                isinstance(value, str)
                and isinstance(current_value, int)
                and is_number(value, int)
            ):
                value = int(value)
            if type(value) != type(current_value):
                raise ValueError(
                    "key '{}' has incorrect type (expected {}, found {})".format(
                        key, type(current_value), type(value)
                    )
                )
            if overwrite == Config.Overwrite.No:
                return current_value
            if overwrite == Config.Overwrite.Error and value != current_value:
                raise ValueError("key '{}' cannot be overwritten".format(key))

        # all fine, set value
        data[splits[-1]] = value
        if log:
            self.log(
                "Set {}={} (was {})".format(
                    key,
                    repr(value),
                    repr(current_value) if current_value is not None else "unset",
                )
            )
        return value

    def set_all(
        self, new_options: Dict[str, Any], create=False, overwrite=Overwrite.Yes
    ):
        for key, value in Config.flatten(new_options).items():
            self.set(key, value, create, overwrite)

    def load(
        self,
        filename: str,
        create=False,
        overwrite=Overwrite.Yes,
        allow_deprecated=True,
    ):
        """Update configuration options from the specified YAML file.
        All options that do not occur in the specified file are retained.
        If ``create`` is ``False``, raises :class:`ValueError` when the file
        contains a non-existing options. When ``create`` is ``True``, allows
        to add options that are not present in this configuration.
        If the file has an import or model field, the corresponding
        configuration files are imported.
        """
        with open(filename, "r") as file:
            new_options = yaml.load(file, Loader=yaml.SafeLoader)
        if new_options is not None:
            self.load_options(
                new_options,
                create=create,
                overwrite=overwrite,
                allow_deprecated=allow_deprecated,
            )

    def load_options(
        self, new_options, create=False, overwrite=Overwrite.Yes, allow_deprecated=True
    ):
        "Like `load`, but loads from an options object obtained from `yaml.load`."

        # check for modules first, so if it's necessary we can import from them.
        if "modules" in new_options:
            modules = set(self.options.get("modules", []))
            modules = modules.union(new_options.get("modules"))
            self.set("modules", list(modules), create=True)
            del new_options["modules"]

        # import model configurations
        if "model" in new_options:
            model = new_options.get("model")
            # TODO not sure why this can be empty when resuming an ax
            # search with model as a search parameter
            if model:
                self._import(model)

        # import explicit imports
        if "import" in new_options:
            imports = new_options.get("import")
            if not isinstance(imports, list):
                imports = [imports]
            for module_name in imports:
                self._import(module_name)
            del new_options["import"]

        # now set all options
        self.set_all(new_options, create, overwrite)

    def load_config(
        self, config, create=False, overwrite=Overwrite.Yes, allow_deprecated=True
    ):
        "Like `load`, but loads from a Config object."
        self.load_options(config.options, create, overwrite, allow_deprecated)

    def save(self, filename):
        """Save this configuration to the given file"""
        with open(filename, "w+") as file:
            file.write(yaml.dump(self.options))

    def save_to(self, checkpoint: Dict) -> Dict:
        """Adds the config file to a checkpoint"""
        checkpoint["config"] = self
        return checkpoint

    @staticmethod
    def flatten(options: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a dictionary of flattened configuration options."""
        result = {}
        Config.__flatten(options, result)
        return result

    @staticmethod
    def __flatten(options: Dict[str, Any], result: Dict[str, Any], prefix=""):
        for key, value in options.items():
            fullkey = key if prefix == "" else prefix + "." + key
            if type(value) is dict:
                Config.__flatten(value, result, prefix=fullkey)
            else:
                result[fullkey] = value

    def clone(self, subfolder: str = None) -> "Config":
        """Return a deep copy"""
        new_config = Config(folder=copy.deepcopy(self.folder), load_default=False)
        new_config.options = copy.deepcopy(self.options)
        if subfolder is not None:
            new_config.folder = os.path.join(self.folder, subfolder)
        return new_config

    def _import(self, module_name: str):
        """Imports the specified module configuration.
        Adds the configuration options from <module_name>.yaml to
        the configuration. Retains existing module configurations, but verifies
        that fields and their types are correct.
        """

        # load the module_name
        module_config = Config(load_default=False)

        # add the importing config's modules to the imported config
        module_names = self.get_default("modules")
        module_config.set("modules", module_names, create=True)

        from esmart.misc import filename_in_module

        config_filename = filename_in_module(self.modules(), f"{module_name}.yaml")
        module_config.load(config_filename, create=True)

        if "import" in module_config.options:
            del module_config.options["import"]

        # add/verify current configuration
        for key in module_config.options.keys():
            cur_value = None
            try:
                cur_value = {key: self.get(key)}
            except KeyError:
                continue
            module_config.set_all(cur_value, create=False)

        # now update this configuration
        self.set_all(module_config.options, create=True)

        # remember the import
        imports = self.options.get("import")
        if imports is None:
            imports = module_name
        elif isinstance(imports, str):
            imports = [imports, module_name]
        else:
            imports.append(module_name)
            imports = list(dict.fromkeys(imports))
        self.options["import"] = imports

    def modules(self) -> List[types.ModuleType]:
        import importlib

        return [importlib.import_module(m) for m in self.get("modules")]


    def checkpoint_file(self, cpt_id: Union[str, int]) -> str:
        "Return path of checkpoint file for given checkpoint id"
        from esmart.misc import is_number

        if is_number(cpt_id, int):
            return os.path.join(self.folder, "checkpoint_{:05d}.pt".format(int(cpt_id)))
        else:
            return os.path.join(self.folder, "checkpoint_{}.pt".format(cpt_id))


    def last_checkpoint_number(self) -> Optional[int]:
        "Return number (epoch) of latest checkpoint"
        found_epoch = -1
        for f in os.listdir(self.folder):
            if re.match("checkpoint_\d{5}\.pt", f):
                new_found_epoch = int(f.split("_")[1].split(".")[0])
                if new_found_epoch > found_epoch:
                    found_epoch = new_found_epoch
        if found_epoch >= 0:
            return found_epoch
        else:
            return None
    # -- CONVENIENCE METHODS --------------------------------------------------

    def _check(self, key: str, value, allowed_values) -> Any:
        if value not in allowed_values:
            raise ValueError(
                "Illegal value {} for key {}; allowed values are {}".format(
                    value, key, allowed_values
                )
            )
        return value

    def check(self, key: str, allowed_values) -> Any:
        """Raise an error if value of key is not in allowed.
        If fine, returns value.
        """
        return self._check(key, self.get(key), allowed_values)

    # -- LOGGING AND TRACING -----------------------------------------------------------

    def log(self, msg: str, echo=True, prefix=""):
        """Add a message to the default log file.
        Optionally also print on console. ``prefix`` is used to indent each
        output line.
        """
        with open(self.logfile(), "a") as file:
            for line in msg.splitlines():
                if prefix:
                    line = prefix + line
                if self.log_prefix:
                    line = self.log_prefix + line
                if echo:
                    self.print(line)
                file.write(str(datetime.datetime.now()) + " " + line + "\n")

    def print(self, *args, **kwargs):
        "Prints the given message unless console output is disabled"
        if not self.exists("console.quiet") or not self.get("console.quiet"):
            print(*args, **kwargs)

    def trace(
        self, echo=False, echo_prefix="", echo_flow=False, log=False, **kwargs
    ) -> Dict[str, Any]:
        """Write a set of key-value pairs to the trace file.
        The pairs are written as a single-line YAML record. Optionally, also
        echo to console and/or write to log file.
        And id and the current time is automatically added using key ``timestamp``.
        Returns the written k/v pairs.
        """
        kwargs["timestamp"] = time.time()
        kwargs["entry_id"] = str(uuid.uuid4())

        ## TODO: fix this problems
        for key in kwargs.keys():
            if isinstance(kwargs[key], np.float32):
                kwargs[key] = float(kwargs[key])
            elif isinstance(kwargs[key], np.int64):
                kwargs[key] = int(kwargs[key])

        line = yaml.dump(kwargs, width=float("inf"), default_flow_style=False).strip()
        if echo or log:
            msg = yaml.dump(kwargs, default_flow_style=echo_flow)
            if log:
                self.log(msg, echo, echo_prefix)
            else:
                for line in msg.splitlines():
                    if echo_prefix:
                        line = echo_prefix + line
                        self.print(line)
        with open(self.tracefile(), "a") as file:
            file.write(line + "\n")
        return kwargs

    # -- FOLDERS AND CHECKPOINTS ----------------------------------------------

    def init_folder(self):
        """Initialize the output folder.
        If the folder does not exists, create it, dump the configuration
        there and return ``True``. Else do nothing and return ``False``.
        """
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            os.makedirs(os.path.join(self.folder, "config"))
            self.save(os.path.join(self.folder, "config.yaml"))
            return True
        return False

    @staticmethod
    def create_from(checkpoint: Dict) -> Config:
        """Create a config from a checkpoint."""
        config = Config()  # round trip to handle deprecated configs
        if "config" in checkpoint and checkpoint["config"] is not None:
            config.load_config(checkpoint["config"].clone())
        if "folder" in checkpoint and checkpoint["folder"] is not None:
            config.folder = checkpoint["folder"]
        return config

    @staticmethod
    def from_options(options: Dict[str, Any] = {}, **more_options) -> Config:
        """Convert given options or kwargs to a Config object.
        Does not perform any checks for correctness."""
        config = Config(load_default=False)
        config.set_all(options, create=True)
        config.set_all(more_options, create=True)
        return config

    def checkpoint_file(self, cpt_id: Union[str, int]) -> str:
        "Return path of checkpoint file for given checkpoint id"
        from esmart.misc import is_number

        if is_number(cpt_id, int):
            return os.path.join(self.folder, "checkpoint_{:05d}.pt".format(int(cpt_id)))
        else:
            return os.path.join(self.folder, "checkpoint_{}.pt".format(cpt_id))

    def last_checkpoint_number(self) -> Optional[int]:
        "Return number (epoch) of latest checkpoint"
        found_epoch = -1
        for f in os.listdir(self.folder):
            if re.match("checkpoint_\d{5}\.pt", f):
                new_found_epoch = int(f.split("_")[1].split(".")[0])
                if new_found_epoch > found_epoch:
                    found_epoch = new_found_epoch
        if found_epoch >= 0:
            return found_epoch
        else:
            return None

    @staticmethod
    def best_or_last_checkpoint_file(path: str) -> str:
        """Return best (if present) or last checkpoint path for a given folder path."""
        if not os.path.exists(path):
            raise Exception("Path or file {} does not exist".format(path))

        config = Config(folder=path, load_default=False)
        checkpoint_file = config.checkpoint_file("best")
        if os.path.isfile(checkpoint_file):
            return checkpoint_file
        cpt_epoch = config.last_checkpoint_number()
        if cpt_epoch:
            return config.checkpoint_file(cpt_epoch)
        else:
            raise Exception("Could not find checkpoint in {}".format(path))


    def logfile(self) -> str:
        folder = self.log_folder if self.log_folder else self.folder
        if folder:
            return os.path.join(folder, "esmart.log")
        else:
            return os.devnull

    def tracefile(self) -> str:
        folder = self.log_folder if self.log_folder else self.folder
        if folder:
            return os.path.join(folder, "trace.yaml")
        else:
            return os.devnull

class Configurable:
    """Mix-in class for adding configurations to objects.
    Each configured object has access to a `config` and a `configuration_key` that
    indicates where the object's options can be found in `config`.
    """

    def __init__(self, config: Config, configuration_key: str = None):
        self._init_configuration(config, configuration_key)

    def has_option(self, name: str) -> bool:
        try:
            self.get_option(name)
            return True
        except KeyError:
            return False

    def get_option(self, name: str) -> Any:
        if self.configuration_key:
            return self.config.get_default(self.configuration_key + "." + name)
        else:
            self.config.get_default(name)

    def check_option(self, name: str, allowed_values) -> Any:
        if self.configuration_key:
            return self.config.check_default(
                self.configuration_key + "." + name, allowed_values
            )
        else:
            return self.config.check_default(name, allowed_values)

    def set_option(
        self, name: str, value, create=False, overwrite=Config.Overwrite.Yes, log=False
    ) -> Any:
        if self.configuration_key:
            return self.config.set(
                self.configuration_key + "." + name,
                value,
                create=create,
                overwrite=overwrite,
                log=log,
            )
        else:
            return self.config.set(
                name, value, create=create, overwrite=overwrite, log=log
            )

    def _init_configuration(self, config: Config, configuration_key: Optional[str]):
        r"""Initializes `self.config` and `self.configuration_key`.
        Only after this method has been called, `get_option`, `check_option`, and
        `set_option` should be used. This method is automatically called in the
        constructor of this class, but can also be called by subclasses before calling
        the superclass constructor to allow access to these three methods. May also be
        overridden by subclasses to perform additional configuration.
        """
        self.config = config
        self.configuration_key = configuration_key