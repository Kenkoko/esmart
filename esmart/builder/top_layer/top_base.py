
from esmart import Config, Dataset, Configurable
from esmart.misc import init_from

class TopLayer(Configurable):
    def __init__(
        self, 
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False) -> None:

        Configurable.__init__(self, config, configuration_key)
        self.dataset = dataset
        
    def build(self, input_layer):
        raise NotImplementedError

    @staticmethod
    def create(
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        init_for_load_only=False,
    ) -> "TopLayer":
        """Factory method for top-layer creation."""

        try:
            top_layer_type = config.get_default(configuration_key + ".type")
            class_name = config.get(top_layer_type + ".class_name")
        except:
            raise Exception("Can't find {}.type in config".format(configuration_key))

        try:
            top_layer = init_from(
                class_name,
                config.get("modules"),
                config,
                dataset,
                configuration_key,
                init_for_load_only=init_for_load_only,
            )
            return top_layer
        except:
            config.log(
                f"Failed to create top-layer {top_layer_type} (class {class_name})."
            )
            raise