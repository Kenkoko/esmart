import tensorflow as tf
from esmart import Config, Dataset
from esmart.builder.builder import BaseBuilder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from esmart.builder.esmart_model import EsmartModel

class EfficientNetBuilder(BaseBuilder):
    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        configuration_key: str,
        init_for_load_only=False,
    ) -> None:
        super().__init__(config, dataset, configuration_key, init_for_load_only=init_for_load_only)

        def load_pretraind_efficient_net(name: str):
            #TODO: IMG_SIZE_BEFORE_RCROP
            self.config.log(f"load_pretraind_efficient_net {name}")
            if name  == 'B0': 
                from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0 as EfficientNet
                self.image_size            = 224
                # IMG_SIZE_BEFORE_RCROP = 240
            elif name  == 'B1': 
                from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B1 as EfficientNet
                self.image_size            = 240
                # IMG_SIZE_BEFORE_RCROP = 257
            elif name  == 'B2': 
                from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B2 as EfficientNet
                self.image_size            = 260
                # IMG_SIZE_BEFORE_RCROP = 278
            elif name  == 'B3': 
                from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3 as EfficientNet
                self.image_size            = 300
                # IMG_SIZE_BEFORE_RCROP = 321
            # elif name  == 'B4': 
            #     from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B4 as EfficientNet
            #     self.image_size            = 380
            #     # IMG_SIZE_BEFORE_RCROP = 407
            # elif name  == 'B5': 
            #     from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B5 as EfficientNet
            #     self.image_size            = 456
            # elif name  == 'B6': 
            #     from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B6 as EfficientNet
            #     self.image_size            = 528
            # elif name  == 'B7': 
            #     from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B7 as EfficientNet 
            #     self.image_size            = 600
            else:
                raise ValueError(f'Not found version {name}')
            self.EfficientNet = EfficientNet
        self.efficient_net_ver = self.get_option('efficient_net_ver')
        load_pretraind_efficient_net(self.efficient_net_ver)
        self.img_channels = self.get_option('img_channels')
        self.shape = (None, None, self.img_channels,)
        self.augmentation = self.get_option('augmentation')
        self.dropout = self.get_option('dropout')
        self.initial_weight = self.get_option('initial_weight')


    def build_model(self, weight: None) -> tf.keras.Model:
        inputs = layers.Input(shape=self.shape)
        #TODO: this
        img_augmentation = Sequential(
            [
                preprocessing.RandomFlip(mode='horizontal'),
                preprocessing.RandomRotation(factor=0.15),
                preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                preprocessing.RandomContrast(factor=0.1),
                preprocessing.RandomCrop(self.image_size, self.image_size),
            ],
            name="img_augmentation",
        )   

        x = img_augmentation(inputs) if self.augmentation else inputs
            
        
        model = self.EfficientNet(include_top=False, input_tensor=x, weights=self.initial_weight)
        
        # Freeze the pretrained weights
        model.trainable = False

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)

        x = layers.Dropout(self.dropout, name="top_dropout")(x)
        outputs = layers.Dense(
            self.dataset.get_option('data_arg.num_classes'), 
            activation="softmax", 
            name="pred")(x)

        # Compile
        model = EsmartModel(inputs, outputs, name="EfficientNet")
        # model = tf.keras.Model(inputs, outputs, name="EfficientNet")
        if weight:
            model.set_weights(weight)
        return model

