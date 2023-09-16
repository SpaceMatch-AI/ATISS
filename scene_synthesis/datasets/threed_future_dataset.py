#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

import json
import pickle
import numpy as np
from .utils import parse_threed_future_models
from functools import lru_cache


class ThreedFutureDataset(object):
    def __init__(self, objects):
        assert len(objects) > 0
        self.objects = objects
        self.sizes = None

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(len(self))

    def __getitem__(self, idx):
        return self.objects[idx]

    def get_sizes(self, sizes_path):
        with open(sizes_path, "r") as f:
            self.sizes = json.load(f)

    def _filter_objects_by_label(self, label):
        return [oi for oi in self.objects if oi.label == label]

    @lru_cache(maxsize=None)
    def compute_size(self, jid):
        return np.array(self.sizes[jid])

    def get_closest_furniture_to_box(self, query_label, query_size, topk=1):
        objects = self._filter_objects_by_label(query_label)
        mses = {}
        for i, oi in enumerate(objects):
            oi_size = self.compute_size(oi.model_jid) if self.sizes else oi.size
            oi_size = oi._transform(oi_size)
            oi_size = np.array(
                [
                    np.sqrt(np.sum((oi_size[4] - oi_size[0]) ** 2)) / 2,
                    np.sqrt(np.sum((oi_size[2] - oi_size[0]) ** 2)) / 2,
                    np.sqrt(np.sum((oi_size[1] - oi_size[0]) ** 2)) / 2,
                ]
            )
            mses[oi] = np.sum((oi_size - query_size) ** 2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[:topk]

    def get_closest_furniture_to_2dbox(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = (oi.size[0] - query_size[0]) ** 2 + (
                oi.size[2] - query_size[1]
            ) ** 2
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[0]

    @classmethod
    def from_dataset_directory(
        cls, dataset_directory, path_to_model_info, path_to_models
    ):
        objects = parse_threed_future_models(
            dataset_directory, path_to_models, path_to_model_info
        )
        return cls(objects)

    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)
        return dataset
