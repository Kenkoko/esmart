import tensorflow as tf
from esmart import Config, Dataset, Configurable
from typing import Any, Dict, List, Optional, Union, Tuple
from esmart.misc import init_from

class BaseBuilder(tf.keras.Model, Configurable):
    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        configuration_key=None,
        init_for_load_only=False,
    ) -> None:
        tf.keras.Model.__init__(self)
        Configurable.__init__(self, config, configuration_key)
        self.dataset = dataset

    def build_model(self) -> tf.keras.Model:
        raise NotImplemented

    @staticmethod
    def create(
        config: Config,
        dataset: Dataset,
        configuration_key: Optional[str] = None,
        init_for_load_only=False,
    ) -> "BaseBuilder":
        """Factory method for builder creation."""
        try:
            if configuration_key is not None:
                builder_name = config.get(configuration_key + ".type")
            else:
                builder_name = config.get("builder")
            config._import(builder_name)
            class_name = config.get(builder_name + ".class_name")
        except:
            raise Exception("Can't find {}.type in config".format(configuration_key))

        try:
            builder = init_from(
                class_name,
                config.get("modules"),
                config=config,
                dataset=dataset,
                configuration_key=builder_name,
                init_for_load_only=init_for_load_only,
            )
            # model.to(config.get("job.device"))
            return builder
        except:
            config.log(f"Failed to create model {builder_name} (class {class_name}).")
            raise

