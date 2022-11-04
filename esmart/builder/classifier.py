
import os
from esmart.augmentation.augmentor import Augmentor
from esmart.builder.builder import BaseBuilder
from esmart import Config, Dataset
import importlib
from tensorflow.keras import layers, models
import tensorflow as tf
from esmart.builder.top_layer.top_base import TopLayer


class ClassifierBuilder(BaseBuilder):
    IMG_SIZE = {
        # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#:~:text=for%20each%20model%3A-,Base%20model,-resolution
        'efficientnet': {
            'b0': 224,
            'b1': 240,
            'b2': 260,
            'b3': 300,
            'b4': 380,
            'b5': 456,
            'b6': 528,
            'b7': 600,
        },
        # https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/efficientnet_v2.py#L1264-L1291
        'efficientnet_v2': {
            's': 384,
            'm': 480,
            'l': 480,
        },
    }
    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        configuration_key=None, 
        init_for_load_only=False) -> None:
        
        super().__init__(config, dataset, configuration_key, init_for_load_only)

        self.config.log(f"Classifier init")
        # backbone layer
        pretrain_model = self.check_option('pretrain_model', ['', 'efficientnet', 'efficientnet_v2', 'resnet', 'convnext'])
        self.backbone = None
        if pretrain_model in ['resnet', 'convnext']:
            raise NotImplementedError
        elif pretrain_model == 'efficientnet_v2':
            version = self.check_option('version', ['b0', 'b1', 'b2', 'b3', 's', 'm', 'l'])
            model_name = 'EfficientNetV2{}'.format(version.upper())
            self.img_size = self.IMG_SIZE['efficientnet'][version]
            module = 'tensorflow.keras.applications.efficientnet_v2'
        elif pretrain_model == 'efficientnet':
            version = self.check_option('version', ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
            model_name = 'EfficientNet{}'.format(version.upper())
            self.img_size = self.IMG_SIZE['efficientnet'][version]
            module = 'tensorflow.keras.applications.efficientnet'
        else:
            raise ValueError('pretrain_model must be in [none, efficientnet, resnet, convnext]')
        if pretrain_model != '':
            self.backbone = getattr(importlib.import_module(module), model_name)
            self.model_name = model_name

        # hyperparameters
        self.img_channels = self.get_option('img_channels')
        self.regularize = self.check_option('regularize.type', ['', 'l1', 'l2', 'l1_l2'])
        if self.get_option('img_size') != -1:
            self.img_size = self.get_option('img_size')

        self.config.log(f"Image size: {self.img_size}")
        self.config.set(f'{self.configuration_key}.img_size', self.img_size, create=True)
        self.config.save(os.path.join(self.config.folder, "config.yaml"))

        # augmentations
        self.augmentations = self.get_option('augmentation')
        self.custom_objects = {}
        if self.augmentations:
            augmentor: Augmentor = Augmentor(self.config, self.augmentations)
            self.config.log(f"Augmentations: {augmentor.augmentations}")
            self.augmentations = augmentor.get_augmentations()
            self.custom_objects.update(augmentor.custom_augmentations)
        else:
            self.augmentations = []
        
        # input alignment layer
        train_img_size = self.config.get(f"{self.config.get('image_processor')}.training.size")
        if train_img_size == -1:
            train_img_size = self.img_size
            self.config.set(f"{self.config.get('image_processor')}.training.size", self.img_size, create=True)
            self.config.save(os.path.join(self.config.folder, "config.yaml"))

        if train_img_size > self.img_size:
            self.config.log(f"Training image size ({train_img_size}x{train_img_size}) is larger than classifier input shape ({self.img_size}x{self.img_size}).\n Random crop is applied.")
            self.augmentations.append(
                layers.experimental.preprocessing.RandomCrop(self.img_size, self.img_size)
            )
        elif train_img_size < self.img_size:
            raise ValueError(f"Training image size ({train_img_size}x{train_img_size}) is smaller than classifier input shape ({self.img_size}x{self.img_size}).")
        

        # create the augmentation layer
        if self.augmentations != []:
            self.augmentations = models.Sequential(self.augmentations, name="img_augmentation")
        else:
            self.augmentations = None

        # top layer
        self.top_layer: TopLayer = TopLayer.create(
            config,
            dataset,
            self.configuration_key + ".top_layer",
            init_for_load_only=init_for_load_only,
        )


    def build_model(self, weight=None) -> tf.keras.Model:
        inputs = layers.Input(shape=(None, None, self.img_channels))

        # augmentations
        x = self.augmentations(inputs) if self.augmentations else inputs
        
        # backbone
        if self.backbone:
            model = self.backbone(include_top=False, input_tensor=x)
            model.trainable = False
            self.config.log(f"Backbone: {self.backbone.__name__}")
            
            x = model.output

        # Rebuild top
        outputs = self.top_layer.build(x)

        # Compile
        model = tf.keras.Model(inputs, outputs, name=self.model_name)
        # adding regularization to all layers
        if not self.regularize == '':
            penalty = self.get_option('regularize.penalty')
            self.config.log(f"Adding regularization {self.regularize} - penalty {penalty} to all layers")
            regularizer = getattr(
                importlib.import_module('tensorflow.keras.regularizers'), 
                self.regularize
                )(penalty)
            model = BaseBuilder.add_regularization(
                model, regularizer,
                custom_objects=self.custom_objects if self.custom_objects else None,
                except_layers=['pred'])

        # save the model summary
        model.summary(print_fn=self.config.log)
        self.config.log(f'Number of non-trainable variables = {len(model.non_trainable_weights)}')
        self.config.log(f'Number of trainable variables = {len(model.trainable_weights)} - {model.trainable_weights}')
        self.config.log(f'Number of variables = {len(model.weights)}')

        if weight:
            model.set_weights(weight)
        return model