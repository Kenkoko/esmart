from esmart.config import Config
from esmart.processor.processor import BaseProcessor
import tensorflow as tf
from functools import partial

class SimpleProcessor(BaseProcessor):
    def __init__(self, 
        config: Config, 
        configuration_key: str = None, 
        init_for_load_only=False,
    ) -> None:
        super().__init__(config, configuration_key, init_for_load_only)
        self.num_classes = self.config.get('dataset.data_arg.num_classes')


        builder_name = self.config.get('builder')
        self.img_channels = self.config.get(f'{builder_name}.img_channels')
        self.valid_img_size = self.config.get(f'{builder_name}.img_size')


        self.train_img_size = self.get_option('training.size')
        self.train_img_size = self.valid_img_size if self.train_img_size == -1 else self.train_img_size
        
        self.train_resize_method = self.get_option('training.method')
        self.valid_resize_method = self.get_option('inference.method')

    @staticmethod
    def processor(file_data, label=None, print_fun=print, img_channels=None, img_size=None, resize_method=None, num_classes=None):

        """Create a processor for a single image."""
        try:
            image_decoded = tf.image.decode_jpeg(
                tf.io.read_file(file_data), channels=img_channels)
        except BaseException as e:
            print_fun(
                f"Aborting loading due to failure of loading file {file_data}"
            )
            raise e

        # resizing image
        print_fun(f'resize images by {resize_method} to {img_size} x {img_size} for training dataset')

        ## get the resizing function
        resize_func = getattr(tf.image, resize_method)
        if resize_method == 'resize':
            image_decoded = resize_func(image_decoded, (img_size, img_size))
        elif resize_method == 'resize_with_pad':
            image_decoded = resize_func(image_decoded, img_size, img_size)
        else:
            raise ValueError(f'Unknown resize method {resize_method}')

        # encoding labels
        if label is not None:
            label = tf.one_hot(label, num_classes)
            return image_decoded, label
        else:
            return image_decoded

    def get_moap_preprocessor(self):
        output = []
        for method in [self.train_resize_method, self.valid_resize_method]:
            if method == 'resize':
                output.append('raw_preprocessing_multi_crops_tensorflow')
            elif method == 'resize_with_pad':
                output.append('raw_preprocessing_multi_crops_tensorflow_with_pad')
            else:
                raise ValueError(f'Unknown resize method {method}')
        return {
            'train_preprocessor': output[0],
            'valid_preprocessor': output[1],
        }

    def get_processor(self, context: str) :
        if context == "train":
            return partial(
                SimpleProcessor.processor, 
                print_fun=self.config.log, 
                img_channels=self.img_channels, 
                img_size=self.train_img_size, 
                resize_method=self.train_resize_method, 
                num_classes=self.num_classes)
        elif context == "valid":
            return partial(
                SimpleProcessor.processor, 
                print_fun=self.config.log, 
                img_channels=self.img_channels, 
                img_size=self.valid_img_size, 
                resize_method=self.valid_resize_method, 
                num_classes=self.num_classes)
        else:
            raise Exception("Unknown context: {}".format(context))