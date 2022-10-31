
from esmart.builder.builder import BaseBuilder
from esmart import Config, Dataset
import importlib
from tensorflow.keras import layers

class ClassifierAttention(BaseBuilder):
    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        configuration_key=None, 
        init_for_load_only=False) -> None:
        
        super().__init__(config, dataset, configuration_key, init_for_load_only)

        self.config.log(f"ClassifierAttention init")
        # backbone layer
        pretrain_model = self.check_option('pretrain_model', ['efficientnet', 'resnet', 'convnext'])
        if pretrain_model in ['resnet', 'convnext']:
            raise NotImplementedError
        elif pretrain_model == 'efficientnet':
            version = self.check_option('version', ['b0', 'b1', 'b2', 'b3',])
            model_name = 'EfficientNetV2{}'.format(version.upper())
            module = 'tensorflow.keras.applications.efficientnet_v2'
        else:
            raise ValueError('pretrain_model must be in [efficientnet, resnet, convnext]')
        self.backbone = getattr(importlib.import_module(module), model_name)

        # attention layer
        self.num_heads = self.get_option('attention.num_heads')
        self.key_dim = self.get_option('attention.key_dim')
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim)
        
        # embedding layer
        self.dim = self.get_option('position_embedder.dim')
        self.num_patches = self.get_option('position_embedder.num_patches')
        self.regularize = self.check_option('regularize', ['', 'l1', 'l2', 'l1_l2'])
        self.embedding = layers.Embedding(
            input_dim=self.num_patches, 
            output_dim=self.dim,
            embeddings_regularizer=self.regularize,
            name='patch_embedding',)


    def build(self):
        pass