from esmart import Config, Dataset, Configurable
from esmart.builder.input_layer.input_base import InputLayer
from tensorflow.keras import layers

class NormalInputLayer(InputLayer):

    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        configuration_key=None, 
        init_for_load_only=False) -> None:
        super().__init__(config, dataset, configuration_key, init_for_load_only)
        self.img_channels = self.get_option('img_channels')

    def build(self):
        outputs = layers.Input(shape=(None, None, self.img_channels))
        return outputs