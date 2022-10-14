from typing import Optional
from esmart.config import Config, Configurable
from esmart.dataset import Dataset
from esmart.misc import init_from


class BaseProcessor(Configurable):
    
    def __init__(
        self, 
        config: Config, 
        configuration_key: str = None, 
        init_for_load_only=False,
    ) -> None:
        super().__init__(config, configuration_key)

    def get_moap_preprocessor(self):
        raise NotImplementedError

    def get_processor(self, context: str):
        """Get the processor for a given context. Context should be one of the following: train, validation, test, inference"""
        raise NotImplementedError

    @staticmethod
    def create(
        config: Config, 
        configuration_key: Optional[str] = None, 
        init_for_load_only=False
        ) -> "BaseProcessor":
        """Factory method for processor creation."""
        try:
            if configuration_key is not None:
                processor_name = config.get(configuration_key + ".type")
            else:
                processor_name = config.get("image_processor")
            config._import(processor_name)
            class_name = config.get(processor_name + ".class_name")
        except:
            raise Exception("Can't find {}.type in config".format(configuration_key))
        
        try:
            processor = init_from(
                class_name,
                config.get("modules"),
                config=config,
                configuration_key=processor_name,
                init_for_load_only=init_for_load_only,
            )
            return processor
        except:
            config.log(f"Failed to create processor {processor_name} (class {class_name}).")
            raise