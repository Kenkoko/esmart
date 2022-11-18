import json
from functools import partial
from typing import List

import numpy as np
import tensorflow as tf
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection

from esmart.config import Config
from esmart.processor.processor import BaseProcessor

import os

class Tokenizer(BaseProcessor):
    def __init__(self, 
        config: Config, 
        configuration_key: str = None, 
        init_for_load_only=False,
    ) -> None:
        super().__init__(config, configuration_key, init_for_load_only)
        self.num_classes = self.config.get('dataset.data_arg.num_classes')


        self.maxium_patch = self.get_option('num_patches')
        self.patch_size = self.get_option('patch_size')
        self.iobjects = self.get_option('iobjects')
        builder_name = self.config.get('builder')
        self.img_channels = self.config.get(f'{builder_name}.input_layer.img_channels')
    
    @staticmethod
    def get_lines(json_file: str) -> List:
        """
        Get lines from json file
        Args:
            json_file: path to json file
        Returns:
            list of dict {
                "name": "Conductor in span",
                "color": "#D33115",
                "shape": "polyline",
                "value": "conductor_in_span",
                "polyline": {
                        "0": {
                            "x": 0.5304,
                            "y": 0
                        },
                        "1": {
                            "x": 0.5498,
                            "y": 0.1346
                        },
                        "2": {
                            "x": 0.5545,
                            "y": 0.1663
                        }
                    },
                ...
            }
        """
        # check if this is json file or not 
        if not json_file.endswith('.json'):
            raise ValueError(f"{json_file} is not json file")
        with open(json_file, 'r') as f:
            annotation = json.load(f)
        return annotation['labels']['objects']

    @staticmethod
    def get_vertical_horizontal_lines(resize_w: int, resize_h: int, patch_size:int) -> List:
        vertical_linestrings = []
        for i in range(0, resize_w, patch_size):
            vertical_linestrings.append(LineString([(i, 0), (i, resize_h)]))
        horizontal_linestrings = []
        for i in range(0, resize_h, patch_size):
            horizontal_linestrings.append(LineString([(0, i), (resize_w, i)]))
        return vertical_linestrings, horizontal_linestrings

    @staticmethod
    def get_conductor_linestring(dict_line_list: List, ratio_w, ratio_h, trans_w, trans_h, iobjects: List) -> List:
        """
        Get conductor linestring from dict_line_list
        Args:
            dict_line_list: list of dict
            ratio_w: ratio of width
            ratio_h: ratio of height
            trans_w: translation of width
            trans_h: translation of height
            iobjects: list of interested objects
        Returns:
            list of conductor linestring
        """
        conductor_linestring = []
        for dict_line in dict_line_list:
            if dict_line['value'] not in iobjects:
                continue
            line_coords = []
            for point in dict_line['polyline']:
                x = dict_line['polyline'][point]['x']
                y = dict_line['polyline'][point]['y']
                x = x * ratio_w + trans_w
                y = y * ratio_h + trans_h
                line_coords.append((x, y))
            conductor_linestring.append(LineString(line_coords))
        return conductor_linestring

    @staticmethod
    def get_grid(height: int, width: int, patch_size: int) -> List:
        """
        Get grid from image
        """
        number_of_patches_width = width / patch_size
        number_of_patches_height = height / patch_size

        if not number_of_patches_width.is_integer() or not number_of_patches_height.is_integer():
            raise ValueError(f"number of patches width or height is not integer")

        # convert to int
        number_of_patches_width = int(number_of_patches_width)
        number_of_patches_height = int(number_of_patches_height)

        # print(f"number_of_patches_width: {number_of_patches_width}")
        # print(f"number_of_patches_height: {number_of_patches_height}")
        # print(f'number of patches: {number_of_patches_width * number_of_patches_height}')
        grid = np.zeros((number_of_patches_width * number_of_patches_height, 4))
        for i in range(number_of_patches_width * number_of_patches_height):
            x = i % number_of_patches_width
            y = i // number_of_patches_width
            x_min = x * patch_size
            y_min = y * patch_size
            x_max = min(x_min + patch_size, width)
            y_max = min(y_min + patch_size, height)
            grid[i] = [x_min, y_min, x_max, y_max]
        return grid

    @staticmethod
    def tokenizer(
        imgpath, map_file, 
        iobjects: List,
        patch_size=512, maxium_patch=150, img_channels=3, 
        print_fun=print, num_classes=2):
        """
        Tokenizer for tf.data.Dataset
        Args:
            imgpath: path to image
            map_file: path to map file
            patch_size: size of patch
        """
        # load image
        try:
            # print_fun(f"load image: {imgpath}")
            # if not os.path.exists(bytes.decode(imgpath.numpy())):
            #     raise ValueError(f"{bytes.decode(imgpath.numpy())} is not exist")
            img_array = tf.image.decode_jpeg(
                tf.io.read_file(imgpath), channels=img_channels)
        except BaseException as e:
            print_fun(
                f"Aborting loading due to failure of loading file {imgpath}"
            )
            raise e

        # get lines
        if type(map_file) != str:
            map_file = bytes.decode(map_file.numpy())
        dict_line_list = Tokenizer.get_lines(map_file)
        height, width, _ = img_array.shape

        new_w = ((width // patch_size + 1) if width % patch_size >= patch_size // 2 else width // patch_size) * patch_size
        new_h = ((height // patch_size + 1) if height % patch_size >= patch_size // 2 else height // patch_size) * patch_size

        real_ratio = min(new_w / width, new_h / height)
        resize_with_pad_img = tf.image.resize_with_pad(img_array, new_h, new_w)

        resize_w = resize_with_pad_img.shape[1]
        resize_h = resize_with_pad_img.shape[0]
        assert resize_w % patch_size == 0 and resize_h % patch_size == 0, 'Resized image size WxH: {}x{}'.format(resize_w, resize_h)

        vertical_linestrings, horizontal_linestrings = Tokenizer.get_vertical_horizontal_lines(resize_w, resize_h, patch_size)
        conductor_linestring = Tokenizer.get_conductor_linestring(
            dict_line_list, 
            width * real_ratio, height * real_ratio, 
            (new_w - width * real_ratio) // 2, (new_h - height * real_ratio) // 2, 
            iobjects)
        grids = Tokenizer.get_grid(resize_h, resize_w, patch_size)
        intersection_points = []
        for conductor in conductor_linestring:
            for vertical_line in vertical_linestrings:
                if conductor.intersects(vertical_line):
                    intersection_points.append(conductor.intersection(vertical_line))
            for horizontal_line in horizontal_linestrings:
                if conductor.intersects(horizontal_line):
                    intersection_points.append(conductor.intersection(horizontal_line))
        # including the beginning and end points of conductor
        for conductor in conductor_linestring:
            intersection_points.append(Point(conductor.coords[0]))
            intersection_points.append(Point(conductor.coords[-1]))

        mask = [0] * len(grids)
        for point in intersection_points:
            for idx, grid in enumerate(grids):
                if mask[idx] == 1:
                    continue
                try:
                    if type(point) == Point:
                        if point.x >= grid[0] and point.x <= grid[2] and point.y >= grid[1] and point.y <= grid[3]:
                            mask[idx] = 1
                    elif type(point) == MultiPoint:
                        for p in point:
                            if p.x >= grid[0] and p.x <= grid[2] and p.y >= grid[1] and p.y <= grid[3]:
                                mask[idx] = 1
                    elif type(point) == GeometryCollection:
                        for p in point:
                            if type(point) != Point:
                                continue
                            if p.x >= grid[0] and p.x <= grid[2] and p.y >= grid[1] and p.y <= grid[3]:
                                mask[idx] = 1
                    elif type(point) == LineString:
                        for p in point.coords:
                            if p[0] >= grid[0] and p[0] <= grid[2] and p[1] >= grid[1] and p[1] <= grid[3]:
                                mask[idx] = 1
                    else:
                        raise ValueError(f"point type is not Point or MultiPoint")
                except BaseException as e:
                    print_fun(f"point: {point}")
                    print_fun(f"grid: {grid}")
                    raise e


        # crop and resize
        grid_yxyx = grids[:, [1, 0, 3, 2]]
        p_coords =  grid_yxyx / [[resize_h, resize_w, resize_h, resize_w]]

        img_patches = tf.image.crop_and_resize([resize_with_pad_img], boxes=p_coords, box_indices=[0]*len(grids), crop_size=(patch_size, patch_size))


        mask = tf.constant(mask, dtype=tf.float32)

        # filling 512x512 zeros to have maxium_patch patches
        if len(grids) < maxium_patch:
            img_patches = tf.concat([img_patches, tf.zeros((maxium_patch - len(grids), patch_size, patch_size, img_channels), dtype=tf.float32)], axis=0)
            mask = tf.concat([mask, tf.zeros((maxium_patch - len(grids)), dtype=tf.float32)], axis=0)
        elif len(grids) > maxium_patch:
            img_patches = img_patches[:maxium_patch]
            mask = mask[:maxium_patch]
        
        # one-hot encoding
        mask = tf.one_hot(tf.cast(mask, dtype=tf.int32), depth=num_classes)

        assert img_patches.shape == (maxium_patch, patch_size, patch_size, img_channels), 'img_patches shape: {}'.format(img_patches.shape)
        assert mask.shape == (maxium_patch, num_classes), 'mask_patches shape: {}'.format(mask.shape)

        return img_patches, mask

    def get_moap_preprocessor(self):
        raise NotImplementedError

    def get_processor(self, context: str) :
        if context in ["train", "valid"]:
            return partial(
                Tokenizer.tokenizer, 
                iobjects=self.iobjects,
                patch_size=self.patch_size, 
                maxium_patch=self.maxium_patch, 
                img_channels=self.img_channels, 
                print_fun=self.config.log, 
                num_classes=self.num_classes,
            )
        else:
            raise Exception("Unknown context: {}".format(context))