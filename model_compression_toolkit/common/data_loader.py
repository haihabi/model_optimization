# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
from typing import List, Callable

import numpy as np
import multiprocessing as mp
from multiprocessing import Process

from PIL import Image

FILETYPES = ['jpeg', 'jpg', 'bmp', 'png']


class FillQueue(object):
    def __init__(self, q: mp.Queue, func):
        self.func = func
        self.q = q

    def run(self):
        while True:
            if self.q.qsize() < 32:
                self.q.put(self.func())


class FolderImageLoader(object):
    """

    Class for images loading, processing and retrieving.

    """

    def __init__(self,
                 folder: str,
                 preprocessing: List[Callable],
                 batch_size: int,
                 file_types: List[str] = FILETYPES,
                 is_multiprocessing: bool = False,
                 num_workers: int = 8):

        """ Initialize a FolderImageLoader object.

        Args:
            folder: Path of folder with images to load. The path have to be exist, and have to contain at
            least one image.
            preprocessing: List of functions to use when processing the images before retrieving them.
            batch_size: Number of images to retrieve each sample.
            file_types: Files types to scan in the folder. Default list is :data:`~model_compression_toolkit.common.data_loader.FILETYPES`
            is_multiprocessing:  A boolean flag that enables multiprocessing of data loading.  Default is False.
            num_workers: an integer that represents the number of parallel works that load data in case of multiprocessing.  Default is 4.

        Examples:

            Instantiate a FolderImageLoader using a directory of images, that returns 10 images randomly each time it is sampled:

            >>> image_data_loader = FolderImageLoader('path/to/images/directory', preprocessing=[], batch_size=10)
            >>> images = image_data_loader.sample()

            To preprocess the images before retrieving them, a list of preprocessing methods can be passed:

            >>> image_data_loader = FolderImageLoader('path/to/images/directory', preprocessing=[lambda x: (x-127.5)/127.5], batch_size=10)

            For the FolderImageLoader to scan only specific files extensions, a list of extensions can be passed:

            >>> image_data_loader = FolderImageLoader('path/to/images/directory', preprocessing=[], batch_size=10, file_types=['png'])

        """
        self.is_multiprocessing = is_multiprocessing
        self.num_workers = num_workers
        self.folder = folder
        self.image_list = []
        print(f"Starting Scanning Disk: {self.folder}")
        for root, dirs, files in os.walk(self.folder):
            for file in files:
                file_type = file.split('.')[-1].lower()
                if file_type in file_types:
                    self.image_list.append(os.path.join(root, file))
        self.n_files = len(self.image_list)
        assert self.n_files > 0, f'Folder to load can not be empty.'
        print(f"Finished Disk Scanning: Found {self.n_files} files")
        self.preprocessing = preprocessing
        self.batch_size = batch_size

        if self.is_multiprocessing:
            self.ctx = mp.get_context('spawn')
            self.q = self.ctx.Queue()

            def _sample():
                return self.read_image(self.batch_size, self.image_list, self.preprocessing, self.n_files)

            self.fq = FillQueue(self.q, _sample)
            self.p_list = []
            print("Starting Multiprocessing")
            for _ in range(num_workers):
                p = Process(target=self.fq.run)
                p.start()
                self.p_list.append(p)

    @staticmethod
    def read_image(in_batch_size, in_image_list, in_preprocessing, in_n_files):
        """
        Read batch_size random images from the image_list the FolderImageLoader holds.
        Process them using the preprocessing list that was passed at initialization, and
        prepare it for retrieving.
        """

        index = np.random.randint(0, in_n_files, in_batch_size)
        image_list = []
        for i in index:
            file = in_image_list[i]
            img = np.uint8(np.array(Image.open(file).convert('RGB')))
            for p in in_preprocessing:  # preprocess images
                img = p(img)
            image_list.append(img)
        return np.stack(image_list, axis=0)

    def _sample(self):
        return self.read_image(self.batch_size, self.image_list, self.preprocessing, self.n_files)

    def sample(self):
        """

        Returns: A sample of batch_size images from the folder the FolderImageLoader scanned.

        """
        if self.is_multiprocessing:
            return self.q.get()  # get current data
        else:
            return self._sample()  # read and return data

    def close(self):
        if self.is_multiprocessing:
            for p in self.p_list:
                p.terminate()
                p.join()
                p.close()
