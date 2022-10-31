import importlib
from esmart import Config, Configurable
from esmart.augmentation.random_brightness import RandomBrightness


class Augmentor(Configurable):
    def __init__(
        self, 
        config: Config,
        configuration_key=None
    ) -> None:
        Configurable.__init__(self, config, configuration_key)
        self.augmentations = self.get_option('techniques')
        self.augmentation_layers = []
        self.custom_augmentations = {}
    def get_augmentations(self):
        for augmentation in self.augmentations:
            self.config.log(f"Augmentation: {augmentation}")
            layer_name = 'Random{}'.format(augmentation.capitalize())
            if layer_name == 'RandomBrightness':
                layer = RandomBrightness(
                    factor=self.config.get('{}.factor'.format(augmentation.lower())),
                    value_range=tuple(self.config.get('{}.value_range'.format(augmentation.lower()))),
                )
                self.custom_augmentations[layer_name] = RandomBrightness
            else:
                layer = getattr(importlib.import_module('tensorflow.keras.layers.experimental.preprocessing'), layer_name)
                layer = layer(**self.config.get(augmentation))
            self.config.log(f"Augmentation layer: {layer.get_config()}")
            self.augmentation_layers.append(layer)
        return self.augmentation_layers
        



