
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
        self.augmentations.append(
            layers.experimental.preprocessing.RandomCrop(self.img_size, self.img_size)
        )
        self.augmentations = models.Sequential(self.augmentations, name="img_augmentation")

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

        # list all augmentations
        
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
            self.config.log(f"Adding regularization to all layers")
            penalty = self.get_option('regularize.penalty')
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
        if weight:
            model.set_weights(weight)
        return model