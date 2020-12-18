# -*- coding: utf-8 -*-
# Copyright 2019-present NAVER Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from PIL import Image
from scipy import io

from torchvision import transforms
import torchvision.transforms.functional as tf

import torch
import os

DATA_DIR = './CUB_200_2011'
TRAIN_TXT = './meta/CUB200/train.txt'
TEST_TXT = './meta/CUB200/test.txt'
BBOX_TXT = './meta/CUB200/bbox.txt'


class ImageData(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        image, image_id, bbox = self.dataset[item]
        image = Image.open(image).convert('RGB')

        # image = tf.to_tensor(image)
        if bbox is not None:
            x, y, w, h = bbox
            width, height = image.size
            crop_box = (x, y, min(x+w, width), min(y+h, height))
            image = image.crop(crop_box)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id

    def __len__(self):
        return len(self.dataset)


class Dataset(object):
    def __init__(self, dataset_dir, train_txt, test_txt, bbox_txt=None):
        self.dataset_dir = dataset_dir
        self._bbox = None

        if bbox_txt is not None:
            self._load_bbox(bbox_txt)

        train, num_train_ids = self._load_meta(train_txt)
        test, num_test_ids = self._load_meta(test_txt)

        self.train = train
        self.test = test

        self.num_train_ids = num_train_ids
        self.num_test_ids = num_test_ids

    def _load_bbox(self, bbox_file):
        self._bbox = {}
        with open(bbox_file, 'r') as f:
            for line in f:
                image_id, x, y, w, h = line.strip().split()
                self._bbox[int(image_id)] = [int(float(x)), int(
                    float(y)), int(float(w)), int(float(h))]

    def _load_meta(self, meta_file):
        datasets = []
        prev_label = -1
        num_class_ids = 0
        with open(meta_file, 'r') as f:
            for line_no, line in enumerate(f):
                if line_no == 0:
                    continue

                image_id, label, image_path = line.strip().split()
                if self._bbox is not None:
                    bbox = self._bbox[int(image_id)]
                else:
                    bbox = None
                datasets.append(
                    (os.path.join(self.dataset_dir, image_path), int(label), bbox))
                if prev_label != int(label):
                    num_class_ids += 1
                    prev_label = int(label)

        return datasets, num_class_ids

    def print_stats(self):
        num_total_ids = self.num_train_ids + self.num_test_ids

        num_train_images = len(self.train)
        num_test_images = len(self.test)
        num_total_images = num_train_images + num_test_images

        print("###### Dataset Statistics ######")
        print("+------------------------------+")
        print("| Subset  | #Classes | #Images |")
        print("+------------------------------+")
        print("| Train   |    {:5d} | {:7d} |".format(
            self.num_train_ids, num_train_images))
        print("| Test    |    {:5d} | {:7d} |".format(
            self.num_test_ids, num_test_images))
        print("+------------------------------+")
        print("| Total   |    {:5d} | {:7d} |".format(
            num_total_ids, num_total_images))
        print("+------------------------------+")


CARS_DIR = './CARS_196'
CARS_TRAIN_MAT = './CARS_196/devkit/cars_train_annos.mat'
CARS_TEST_MAT = './CARS_196/devkit/cars_test_annos.mat'


class CARS_Dataset():
    def __init__(self, dataset_dir, train_mat, test_mat):
        self.dataset_dir = dataset_dir
        self._bbox_train = None
        self._bbox_test = None

        train, num_train_ids = self._load_train(train_mat)
        test, num_test_ids = self._load_test(test_mat)

        self.train = train
        self.test = test

        self.num_train_ids = num_train_ids
        self.num_test_ids = num_test_ids

    # TODO: Refactoring with non duplicated codes
    def _load_train(self, meta_file):
        datasets = []
        num_class_ids = 0

        cars_annos_train = io.loadmat(meta_file)
        self._bbox_train = {}

        for d in cars_annos_train['annotations'][0]:
            num_class_ids += 1
            bbox_x1 = d[0][0][0]
            bbox_y1 = d[1][0][0]
            bbox_x2 = d[2][0][0]
            bbox_y2 = d[3][0][0]
            im_path = d[4][0]

            x = min(bbox_x1, bbox_x2)
            y = min(bbox_y1, bbox_y2)
            w = max(bbox_x1, bbox_x2) - x
            h = max(bbox_y1, bbox_y2) - y

            image_id = int(im_path[:5])
            self._bbox_train[int(image_id)] = [int(x), int(y), int(w), int(h)]
            # image_label = str(image_id) + '_train'
            bbox = self._bbox_train[int(image_id)]
            datasets.append(
                (os.path.join(self.dataset_dir, 'cars_train/' + str(im_path)), image_id, bbox))

        return datasets, num_class_ids

    def _load_test(self, meta_file):
        datasets = []
        num_class_ids = 0

        cars_annos_test = io.loadmat(meta_file)
        self._bbox_test = {}

        for d in cars_annos_test['annotations'][0]:
            num_class_ids += 1
            bbox_x1 = d[0][0][0]
            bbox_y1 = d[1][0][0]
            bbox_x2 = d[2][0][0]
            bbox_y2 = d[3][0][0]
            im_path = d[4][0]

            x = min(bbox_x1, bbox_x2)
            y = min(bbox_y1, bbox_y2)
            w = max(bbox_x1, bbox_x2) - x
            h = max(bbox_y1, bbox_y2) - y

            image_id = int(im_path[:5])
            self._bbox_test[int(image_id)] = [int(x), int(y), int(w), int(h)]
            # image_label = str(image_id) + '_test'
            bbox = self._bbox_test[int(image_id)]
            datasets.append(
                (os.path.join(self.dataset_dir, 'cars_test/' + str(im_path)), image_id, bbox))

        return datasets, num_class_ids

    def print_stats(self):
        num_total_ids = self.num_train_ids + self.num_test_ids

        num_train_images = len(self.train)
        num_test_images = len(self.test)
        num_total_images = num_train_images + num_test_images

        print("###### Dataset Statistics ######")
        print("+------------------------------+")
        print("| Subset  | #Classes | #Images |")
        print("+------------------------------+")
        print("| Train   |    {:5d} | {:7d} |".format(
            self.num_train_ids, num_train_images))
        print("| Test    |    {:5d} | {:7d} |".format(
            self.num_test_ids, num_test_images))
        print("+------------------------------+")
        print("| Total   |    {:5d} | {:7d} |".format(
            num_total_ids, num_total_images))
        print("+------------------------------+")


def test_dataset():
    dataset = Dataset(DATA_DIR, TRAIN_TXT, TEST_TXT, BBOX_TXT)
    dataset.print_stats()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_dataset = ImageData(dataset.test, test_transform)
    for x in img_dataset:
        print(x[0].shape)


def test_cars_dataset():
    dataset = CARS_Dataset(CARS_DIR, CARS_TRAIN_MAT, CARS_TEST_MAT)
    dataset.print_stats()
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_dataset = ImageData(dataset.test, test_transform)
    for x in img_dataset:

        print(x[0].shape)


# test_dataset()
