from esmart import Config, Dataset, Configurable
from esmart.builder.top_layer.top_base import TopLayer
from tensorflow.keras import layers

class NormalLayer(TopLayer):
    def build(self, input_layer):
        x = layers.Dropout(self.get_option('dropout_rate'), name='top_dropout_1')(input_layer)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.experimental.SyncBatchNormalization(
            **self.get_option('batch_normalization')
        )(x)

        x = layers.Dropout(self.get_option('dropout_rate'), name="top_dropout_2")(x)
        outputs = layers.Dense(
            self.dataset.get_option('data_arg.num_classes'), 
            activation=self.get_option('activation'),
            name="pred")(x)
        
        return outputs