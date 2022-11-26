# https://github.com/faustomorales/vit-keras
# https://github.com/google-research/vision_transformer#fine-tuning-a-model
# https://huggingface.co/docs/transformers/model_doc/vit
import os
import warnings

import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_addons as tfa
import typing_extensions as tx

from esmart import Config, Dataset
from esmart.augmentation.augmentor import Augmentor
from esmart.builder.builder import BaseBuilder


class ViTBuilder(BaseBuilder):
    BASE_URL = "https://github.com/faustomorales/vit-keras/releases/download/dl"
    ConfigDict = tx.TypedDict(
        "ConfigDict",
        {
            "dropout": float,
            "mlp_dim": int,
            "num_heads": int,
            "num_layers": int,
            "hidden_size": int,
        },
    )

    CONFIG_B: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
    }

    CONFIG_L: ConfigDict = {
        "dropout": 0.1,
        "mlp_dim": 4096,
        "num_heads": 16,
        "num_layers": 24,
        "hidden_size": 1024,
    }

    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        configuration_key=None, 
        init_for_load_only=False) -> None:
        

        super().__init__(config, dataset, configuration_key, init_for_load_only)
        self.config.log(f"ViT init")
        # backbone layer
        self.version = self.check_option('version', ['', "B_16", "B_32", "L_16", "L_32"])
        self.patch_size = int(self.version.split('_')[1])
        if "B" in self.version:
            self.model_config = self.CONFIG_B
            self.representation_size = 768
            self.img_size = 224
        elif "L" in self.version:
            self.model_config = self.CONFIG_L
            self.representation_size = 1024
            self.img_size = 384
        else:
            raise ValueError("version must be in [B_16, B_32, L_16, L_32]")
        
        self.pretrain_model_on = self.check_option('pretrain_model_on', ['', 'imagenet21k', 'imagenet21k+imagenet2012'])
        if self.pretrain_model_on in ['imagenet21k+imagenet2012' , '']:
            self.representation_size = None

        self.num_classes = self.dataset.get_option('data_arg.num_classes')


        # hyperparameters
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
                tf.keras.layers.experimental.preprocessing.RandomCrop(self.img_size, self.img_size)
            )
        elif train_img_size < self.img_size:
            raise ValueError(f"Training image size ({train_img_size}x{train_img_size}) is smaller than classifier input shape ({self.img_size}x{self.img_size}).")
        
        # create the augmentation layer
        if self.augmentations != []:
            self.augmentations = tf.keras.models.Sequential(self.augmentations, name="img_augmentation")
        else:
            self.augmentations = None

    def build_model(self, weight=None) -> tf.keras.Model:
        input_layer = tf.keras.layers.Input(shape=(None, None, 3))
        self.config.log(f"Input shape: {input_layer.shape}")
        # augmentations
        if self.augmentations:
            self.config.log(f"Apply augmentations")
            x = self.augmentations(input_layer)
        else:
            self.config.log(f"Apply nothing")
            x = input_layer
        y = tf.keras.layers.Conv2D(
            filters=self.model_config['hidden_size'],
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="embedding",
        )(x)
        y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2], self.model_config['hidden_size']))(y)
        y = ClassToken(name="class_token")(y)
        y = AddPositionEmbs(name="Transformer/posembed_input")(y)
        for n in range(self.model_config['num_layers']):
            y, _ = TransformerBlock(
                num_heads=self.model_config['num_heads'],
                mlp_dim=self.model_config['mlp_dim'],
                dropout=self.model_config['dropout'],
                name=f"Transformer/encoderblock_{n}",
            )(y)
        y = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="Transformer/encoder_norm"
        )(y)
        y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
        if self.representation_size is not None:
            y = tf.keras.layers.Dense(
                self.representation_size, name="pre_logits", activation="tanh"
            )(y)
        

        y = tf.keras.layers.Dense(self.num_classes, name="head", activation='sigmoid')(y)
        model = tf.keras.models.Model(inputs=input_layer, outputs=y, name=f'vit-{self.version.replace("_", "").lower()}')

        # freeze backbone
        for layer in model.layers:
            if layer.name in [
                # 'embedding', 
                # 'class_token', 'Transformer/posembed_input', 
                "pre_logits", 'head'
            ]:
                self.config.log(f"Unfreeze layer: {layer.name}")
                continue
            layer.trainable = False

        # save the model summary
        model.summary(print_fn=self.config.log)
        self.config.log(f'Number of non-trainable variables = {len(model.non_trainable_weights)}')
        self.config.log(f'Number of trainable variables = {len(model.trainable_weights)}')

        if weight:
            model.set_weights(weight)
        elif self.version == '':
            self.config.log(f"Use random initialized model")
        else:
            self.config.log(f"Use pretrained model")
            ViTBuilder.load_pretrained(
                size=self.version,
                weights=self.pretrain_model_on,
                pretrained_top=False,
                model=model,
                patch_size=self.patch_size,
                img_size=self.img_size,
            )
        return model

    @staticmethod
    def load_pretrained(
        size: str,
        weights: str,
        pretrained_top: bool,
        model: tf.keras.models.Model,
        patch_size: int,
        img_size: int,
    ):
        """Load model weights for a known configuration."""
        fname = f"ViT-{size}_{weights}.npz"
        origin = f"{ViTBuilder.BASE_URL}/{fname}"
        local_filepath = tf.keras.utils.get_file(fname, origin, cache_subdir="weights")
        ViTBuilder.load_weights_numpy(
            model=model,
            params_path=local_filepath,
            pretrained_top=pretrained_top,
            num_x_patches=img_size // patch_size,
            num_y_patches=img_size // patch_size,
        )

    @staticmethod
    def load_weights_numpy(
        model, params_path, pretrained_top, num_x_patches, num_y_patches
    ):
        """Load weights saved using Flax as a numpy array.
        Args:
            model: A Keras model to load the weights into.
            params_path: Filepath to a numpy archive.
            pretrained_top: Whether to load the top layer weights.
            num_x_patches: Number of patches in width of image.
            num_y_patches: Number of patches in height of image.
        """
        params_dict = np.load(
            params_path, allow_pickle=False
        )  # pylint: disable=unexpected-keyword-arg
        source_keys = list(params_dict.keys())
        pre_logits = any(l.name == "pre_logits" for l in model.layers)
        source_keys_used = []
        n_transformers = len(
            set(
                "/".join(k.split("/")[:2])
                for k in source_keys
                if k.startswith("Transformer/encoderblock_")
            )
        )
        n_transformers_out = sum(
            l.name.startswith("Transformer/encoderblock_") for l in model.layers
        )
        assert n_transformers == n_transformers_out, (
            f"Wrong number of transformers ("
            f"{n_transformers_out} in model vs. {n_transformers} in weights)."
        )

        matches = []
        for tidx in range(n_transformers):
            encoder = model.get_layer(f"Transformer/encoderblock_{tidx}")
            source_prefix = f"Transformer/encoderblock_{tidx}"
            matches.extend(
                [
                    {
                        "layer": layer,
                        "keys": [
                            f"{source_prefix}/{norm}/{name}" for name in ["scale", "bias"]
                        ],
                    }
                    for norm, layer in [
                        ("LayerNorm_0", encoder.layernorm1),
                        ("LayerNorm_2", encoder.layernorm2),
                    ]
                ]
                + [
                    {
                        "layer": encoder.mlpblock.get_layer(
                            f"{source_prefix}/Dense_{mlpdense}"
                        ),
                        "keys": [
                            f"{source_prefix}/MlpBlock_3/Dense_{mlpdense}/{name}"
                            for name in ["kernel", "bias"]
                        ],
                    }
                    for mlpdense in [0, 1]
                ]
                + [
                    {
                        "layer": layer,
                        "keys": [
                            f"{source_prefix}/MultiHeadDotProductAttention_1/{attvar}/{name}"
                            for name in ["kernel", "bias"]
                        ],
                        "reshape": True,
                    }
                    for attvar, layer in [
                        ("query", encoder.att.query_dense),
                        ("key", encoder.att.key_dense),
                        ("value", encoder.att.value_dense),
                        ("out", encoder.att.combine_heads),
                    ]
                ]
            )
        for layer_name in ["embedding", "head", "pre_logits"]:
            if layer_name == "head" and not pretrained_top:
                source_keys_used.extend(["head/kernel", "head/bias"])
                continue
            if layer_name == "pre_logits" and not pre_logits:
                continue
            matches.append(
                {
                    "layer": model.get_layer(layer_name),
                    "keys": [f"{layer_name}/{name}" for name in ["kernel", "bias"]],
                }
            )
        matches.append({"layer": model.get_layer("class_token"), "keys": ["cls"]})
        matches.append(
            {
                "layer": model.get_layer("Transformer/encoder_norm"),
                "keys": [f"Transformer/encoder_norm/{name}" for name in ["scale", "bias"]],
            }
        )
        ViTBuilder.apply_embedding_weights(
            target_layer=model.get_layer("Transformer/posembed_input"),
            source_weights=params_dict["Transformer/posembed_input/pos_embedding"],
            num_x_patches=num_x_patches,
            num_y_patches=num_y_patches,
        )
        source_keys_used.append("Transformer/posembed_input/pos_embedding")
        for match in matches:
            source_keys_used.extend(match["keys"])
            source_weights = [params_dict[k] for k in match["keys"]]
            if match.get("reshape", False):
                source_weights = [
                    source.reshape(expected.shape)
                    for source, expected in zip(
                        source_weights, match["layer"].get_weights()
                    )
                ]
            match["layer"].set_weights(source_weights)
        unused = set(source_keys).difference(source_keys_used)
        if unused:
            warnings.warn(f"Did not use the following weights: {unused}", UserWarning)
        target_keys_set = len(source_keys_used)
        target_keys_all = len(model.weights)
        if target_keys_set < target_keys_all:
            warnings.warn(
                f"Only set {target_keys_set} of {target_keys_all} weights.", UserWarning
            )

    @staticmethod
    def apply_embedding_weights(target_layer, source_weights, num_x_patches, num_y_patches):
        """Apply embedding weights to a target layer.
        Args:
            target_layer: The target layer to which weights will
                be applied.
            source_weights: The source weights, which will be
                resized as necessary.
            num_x_patches: Number of patches in width of image.
            num_y_patches: Number of patches in height of image.
        """
        expected_shape = target_layer.weights[0].shape
        if expected_shape != source_weights.shape:
            token, grid = source_weights[0, :1], source_weights[0, 1:]
            sin = int(np.sqrt(grid.shape[0]))
            sout_x = num_x_patches
            sout_y = num_y_patches
            warnings.warn(
                "Resizing position embeddings from " f"{sin}, {sin} to {sout_x}, {sout_y}",
                UserWarning,
            )
            zoom = (sout_y / sin, sout_x / sin, 1)
            grid = sp.ndimage.zoom(grid.reshape(sin, sin, -1), zoom, order=1).reshape(
                sout_x * sout_y, -1
            )
            source_weights = np.concatenate([token, grid], axis=0)[np.newaxis]
        target_layer.set_weights([source_weights])
    

# https://github.com/faustomorales/vit-keras/blob/master/vit_keras/layers.py
@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# pylint: disable=too-many-instance-attributes
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)