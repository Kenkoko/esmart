import tensorflow as tf
from esmart import Config, Dataset
from esmart.builder.builder import BaseBuilder
from tensorflow import keras
from tensorflow.keras import layers


class DummyBuilder(BaseBuilder):
    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        configuration_key: str,
        init_for_load_only=False,
    ) -> None:
        super().__init__(config, dataset, configuration_key, init_for_load_only=init_for_load_only)

        self.image_size = self.get_option('img_size')
        self.img_channels = self.get_option('img_channels')
        self.shape = (self.image_size, self.image_size, self.img_channels,)
        self.augmentation = self.get_option('augmentation')
        self.activation = self.get_option('activation')
        self.dropout = self.get_option('dropout')
    
    def build_model(self, weight) -> tf.keras.Model:
        inputs = keras.Input(shape=self.shape)
        data_augmentation = keras.Sequential(
            [
                ## TODO: configuralize this also
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
            ]
        )
        # Image augmentation block
        x = data_augmentation(inputs) if self.augmentation else inputs

        # Entry block
        x = layers.Rescaling(1.0 / 255)(x)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512, 728]:
            x = layers.Activation(self.activation)(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation(self.activation)(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)

        x = layers.GlobalAveragePooling2D()(x)
        # if num_classes == 2:
        #     final_activation = "sigmoid"
        #     units = 1
        # else:
        #     final_activation = "softmax"
        #     units = num_classes

        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(2, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        if weight:
            model.set_weights(weight)
        return model

