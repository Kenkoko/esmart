import tensorflow as tf
from esmart import Config, Dataset, Configurable
from typing import Any, Dict, List, Optional, Union, Tuple
from esmart.misc import init_from
import os
import tempfile

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

    def build_model(self, weight=None) -> tf.keras.Model:
        raise NotImplementedError
    
    @staticmethod
    def add_regularization(model, regularizer: tf.keras.regularizers.Regularizer, except_layers=[], custom_objects=None) -> tf.keras.Model:
        if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
            print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
            return model
        if custom_objects != None:
            print(custom_objects)
        
        for layer in model.layers:
            if layer.name in except_layers:
                continue
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        # When we change the layers attributes, the change only happens in the model config file
        model_json = model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        model.save_weights(tmp_weights_path)

        # load the model from the config
        model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
        
        # Reload the model weights
        model.load_weights(tmp_weights_path, by_name=True)

        assert model.losses != [], "Model must have losses"
        return model

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
    
