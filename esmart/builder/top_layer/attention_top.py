# https://towardsdatascience.com/understand-and-implement-vision-transformer-with-tensorflow-2-0-f5435769093
# https://keras.io/examples/vision/patch_convnet/
from esmart import Config, Dataset, Configurable
from esmart.builder.top_layer.top_base import TopLayer
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.nn import gelu
from tensorflow.keras import regularizers
class AttentionLayer(TopLayer):
    def __init__(
        self,
        config: Config, dataset: Dataset, 
        configuration_key=None, init_for_load_only=False) -> None:
        super().__init__(config, dataset, configuration_key, init_for_load_only)
        self.num_patches = self.get_option('num_patches')
        self.projection_dim = self.get_option('projection_dim')
        self.transformer_layers = self.get_option('transformer_layers')
        self.num_heads = self.get_option('num_heads')
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]
        self.dropout_rate = self.get_option('dropout_rate')
        self.activation = self.get_option('activation')
    @staticmethod
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def build(self, input_layer):
        x = CollapseBatchDim(num_patches=self.num_patches)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((self.num_patches, -1))(x)
        x = layers.Dense(self.projection_dim)(x)
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(x)
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(**self.get_option('LayerNormalization'))(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=self.dropout_rate
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(**self.get_option('layer_normalization'))(x2)
            # MLP.
            x3 = AttentionLayer.mlp(x3, hidden_units=self.transformer_units, dropout_rate=self.dropout_rate)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])
        representation = layers.LayerNormalization(**self.get_option('layer_normalization'))(encoded_patches)
        outputs = layers.Dense(
            self.dataset.get_option('data_arg.num_classes'),
            activation=self.activation,
            name="pred")(representation)
        
        return outputs


class CollapseBatchDim(layers.Layer):
    def __init__(self, num_patches):
            super(CollapseBatchDim, self).__init__()
            self.num_patches = num_patches
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        dim_1 = tf.shape(inputs)[1]
        dim_2 = tf.shape(inputs)[2]
        dim_3 = tf.shape(inputs)[3]
        return tf.reshape(inputs, (batch_size // self.num_patches, self.num_patches, dim_1, dim_2, dim_3))

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim, embeddings_regularizer= regularizers.L2(0.0001))#L3Regularizer(0.001)
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patch + self.position_embedding(positions)
        return encoded

