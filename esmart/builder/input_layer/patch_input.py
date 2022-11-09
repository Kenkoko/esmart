from esmart import Config, Dataset, Configurable
from esmart.builder.input_layer.input_base import InputLayer
import tensorflow as tf

class PatchInputLayer(InputLayer):

    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        configuration_key=None, 
        init_for_load_only=False) -> None:
        super().__init__(config, dataset, configuration_key, init_for_load_only)

        preprocessor = self.config.get('image_processor')
        self.img_channels = self.get_option('img_channels')
        self.num_patches =  self.config.get(f'{preprocessor}.num_patches')
        self.patch_size =  self.config.get(f'{preprocessor}.patch_size')


    def build(self):
        self.config.log(f"Build patch input layer")
        self.config.log(f"Image channels: {self.img_channels}")
        self.config.log(f"Number of patches: {self.num_patches}")
        self.config.log(f"Patch size: {self.patch_size}")
        input_layer = tf.keras.layers.Input(shape=(self.patch_size, self.num_patches, self.num_patches, self.img_channels))
        outputs = ExpandBatchDim()(input_layer)
        return outputs, input_layer

class ExpandBatchDim(tf.keras.layers.Layer):
  def __init__(self):
        super(ExpandBatchDim, self).__init__()
  def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_patches = tf.shape(inputs)[1]
        dim_1 = tf.shape(inputs)[2]
        dim_2 = tf.shape(inputs)[3]
        dim_3 = tf.shape(inputs)[4]
        return tf.reshape(inputs, (batch_size * num_patches, dim_1, dim_2, dim_3))

