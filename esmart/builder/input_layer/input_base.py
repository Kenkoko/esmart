
from esmart import Config, Dataset, Configurable
from esmart.misc import init_from

class InputLayer(Configurable):
    def __init__(
        self, 
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False) -> None:

        Configurable.__init__(self, config, configuration_key)
        self.dataset = dataset
        
    def build(self):
        raise NotImplementedError

    @staticmethod
    def create(
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        init_for_load_only=False,
    ) -> "InputLayer":
        """Factory method for input-layer creation."""
        print("InputLayer.create")
        print("configuration_key: ", configuration_key)
        print("config: ", config.options)
        try:
            input_layer_type = config.get_default(configuration_key + ".type")
            print("input_layer_type: ", input_layer_type)
            class_name = config.get(input_layer_type + ".class_name")
        except:
            raise Exception("Can't find {}.type in config".format(configuration_key))

        try:
            input_layer = init_from(
                class_name,
                config.get("modules"),
                config,
                dataset,
                configuration_key,
                init_for_load_only=init_for_load_only,
            )
            return input_layer
        except:
            config.log(
                f"Failed to create input-layer {input_layer_type} (class {class_name})."
            )
            raise