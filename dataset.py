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
CARS_MAT = './CARS_196/devkit/cars_annos.mat'


class CARS_Dataset():
    def __init__(self, dataset_dir, cars_mat):
        self.dataset_dir = dataset_dir
        self._bbox_train = None
        self._bbox_test = None

        train, num_train_ids, test, num_test_ids = self._load_meta(cars_mat)

        self.train = train
        self.test = test

        self.num_train_ids = num_train_ids
        self.num_test_ids = num_test_ids

    def _load_meta(self, meta_file):
        test_datasets = []
        train_datasets = []
        prev_test_label = -1
        num_test_class_ids = 0
        prev_train_label = -1
        num_train_class_ids = 0

        cars_annos = io.loadmat(meta_file)
        self._bbox = {}

        for d in cars_annos['annotations'][0]:
            im_path = d[0][0]
            bbox_x1 = d[1][0][0]
            bbox_y1 = d[2][0][0]
            bbox_x2 = d[3][0][0]
            bbox_y2 = d[4][0][0]
            im_class = d[5][0][0]
            im_is_test = d[6][0][0]

            x = min(bbox_x1, bbox_x2)
            y = min(bbox_y1, bbox_y2)
            w = max(bbox_x1, bbox_x2) - x
            h = max(bbox_y1, bbox_y2) - y

            image_id = int(im_path[8:14])
            image_folder = str(int(image_id / 1000))
            if len(image_folder) == 1:
                image_folder = '0' + image_folder
            self._bbox[image_id] = [int(x), int(y), int(w), int(h)]
            bbox = self._bbox[image_id]

            if im_is_test:
                test_datasets.append(
                    (os.path.join(self.dataset_dir, image_folder + im_path[7:]), int(im_class), bbox))
                if prev_test_label != im_class:
                    num_test_class_ids += 1
                    prev_test_label = im_class
            else:
                train_datasets.append(
                    (os.path.join(self.dataset_dir, image_folder + im_path[7:]), int(im_class), bbox))
                if prev_train_label != im_class:
                    num_train_class_ids += 1
                    prev_train_label = im_class

        return train_datasets, num_train_class_ids, test_datasets, num_test_class_ids

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
        print(type(x[1]))



def test_cars_dataset():
    dataset = CARS_Dataset(CARS_DIR, CARS_MAT)
    dataset.print_stats()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_dataset = ImageData(dataset.test, None)
    img_dataset[5555][0].show()



#test_dataset()
#test_cars_dataset()
