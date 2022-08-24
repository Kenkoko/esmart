from __future__ import annotations

import os
import uuid

from esmart import Config, Configurable

import tensorflow as tf
from sklearn.model_selection import train_test_split

from typing import Dict, List, Any, Callable, Union, Optional
from collections import Counter

class Dataset(Configurable):
    """Stores information about a dataset.
    This includes the number of entities, number of relations, splits containing tripels
    (e.g., to train, validate, test), indexes, and various metadata about these objects.
    Most of these objects can be lazy-loaded on first use.
    """
    def __init__(self, config: Config):
        """Constructor for internal use.
        To load a dataset, use `Dataset.create()`."""
        super().__init__(config, "dataset")

        #: directory in which dataset is stored
        self.folder = config.get("dataset.folder")

        self._file_path = {}
        self._labels = {}

        self.num_classes = config.get('dataset.data_arg.num_classes')
        self.class_names = config.get('dataset.data_arg.class_names')

        self.folder_train = os.path.join(self.folder, self.config.get(f"dataset.train.data_dir"))
        self.file_train = os.path.join(self.folder, self.config.get(f"dataset.train.file"))

        self.folder_test = os.path.join(self.folder, self.config.get(f"dataset.test.data_dir"))
        self.file_test = os.path.join(self.folder, self.config.get(f"dataset.test.file"))

        self.folder_valid = os.path.join(self.folder, self.config.get(f"dataset.valid.data_dir"))
        self.file_valid = os.path.join(self.folder, self.config.get(f"dataset.valid.file"))

    def read_filelist(self, file_data, data_dir):
        f = open(file_data, "r")
        lines = f.readlines()
        f.close()
        filepaths = []
        labels = []
        for line in lines:
            parts = line[:-1].split('|||')
            filepaths.append(os.path.join(data_dir, parts[0]))
            labels.append(self.class_names.index(parts[1]))
        self.config.log(f'{file_data} contains {", ".join("{}: {}".format(self.class_names[k], v) for k, v in Counter(labels).most_common())}')
        return filepaths, labels

    def load_filepath(self, key: str):
        "load or return the path file with the specified key"
        if key not in self._file_path:
            if key == 'train':
                #TODO make this more informative
                if self.file_train == '':
                    raise ValueError('Need to input file train and folder path for training dataset')
                self.config.log(f'Loading {key} from {self.file_train}')
                file_paths, labels = self.read_filelist(self.file_train, self.folder_train)
                if self.config.get(f"dataset.valid.data_dir") == '':
                    self.config.log(f"Validation director is None, create valid dataset from training dataset {self.file_train}")
                    X_train, X_valid, y_train, y_valid = train_test_split(
                        file_paths, labels, 
                        test_size=self.config.get('dataset.valid.split_ratio'), 
                        random_state=self.config.get('dataset.random_state'))
                    self._file_path['valid'] = X_valid
                    self._labels['valid'] = y_valid
                else:
                    X_train, y_train = file_paths, labels
                self._file_path['train'] = X_train
                self._labels['train'] = y_train

            if key == 'valid':
                if self.config.get(f"dataset.valid.data_dir") == '':
                    #TODO make this more informative
                    if self.file_train == '':
                        raise ValueError('Need to input file train and folder path for training dataset')
                    self.config.log(f"Validation director is None, create valid dataset from training dataset {self.file_train}")
                    file_paths, labels = self.read_filelist(self.file_train, self.folder_train)
                    X_train, X_valid, y_train, y_valid = train_test_split(
                        file_paths, labels, 
                        test_size=self.config.get('dataset.valid.split_ratio'), 
                        random_state=self.config.get('dataset.random_state'))
                    self._file_path['train'] = X_train
                    self._labels['train'] = y_train
                else:
                    self.config.log(f'Loading {key} from {self.file_valid}')
                    X_valid, y_valid = self.read_filelist(self.file_valid, self.folder_valid)

                self._file_path['valid'] = X_valid
                self._labels['valid'] = y_valid

            if key == 'test':
                if self.file_test == '':
                    raise ValueError('Need to input file train and folder path for training dataset')
                self.config.log(f'Loading {key} from {self.file_test}')
                file_paths, labels = self.read_filelist(self.file_test, self.folder_test)
                self._file_path['test'] = file_paths
                self._labels['test'] = labels
        self.config.log(f'{key} contains {", ".join("{}: {}".format(self.class_names[k], v) for k, v in Counter(self._labels[key]).most_common())}')
        return self._file_path[key], self._labels[key]

    def split(self, split: str):
        """Return the split of the specified name.
        """
        return self.load_filepath(split)

    @staticmethod
    def create(config: Config, preload_data: bool = True):
        """Loads a dataset.
        If preload_data is set, loads entity and relation maps as well as all splits.
        Otherwise, this data is lazy loaded on first use.
        """
        dataset = Dataset(config)

        config.log(f"Loading configuration of dataset from {dataset.folder} ...")
        
        if preload_data:
            for split in ["train", "valid", "test"]:
                dataset.split(split)
        return dataset

    @staticmethod
    def create_from(
        checkpoint: Dict,
        config: Config = None,
        dataset: Optional[Dataset] = None,
        preload_data=False,
    ) -> Dataset:
        """Creates dataset based on a checkpoint.
        If a dataset is provided, only (!) its meta data will be updated with the values
        from the checkpoint. No further checks are performed.
        Args:
            checkpoint: loaded checkpoint
            config: config (should match the one of checkpoint if set)
            dataset: dataset to update
            preload_data: preload data
        Returns: created/updated dataset
        """
        if config is None:
            config = Config.create_from(checkpoint)
        if dataset is None:
            dataset = Dataset.create(config, preload_data)
        if "dataset" in checkpoint:
            dataset_checkpoint = checkpoint["dataset"]
            if (
                "dataset.meta" in dataset_checkpoint
                and dataset_checkpoint["meta"] is not None
            ):
                dataset._meta.update(dataset_checkpoint["meta"])
            # dataset._num_entities = dataset_checkpoint["num_entities"]
            # dataset._num_relations = dataset_checkpoint["num_relations"]
        return dataset