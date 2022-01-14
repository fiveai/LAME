import requests
import tarfile
import os
from os.path import join as ospjoin

# -**-

import shutil
from pathlib import Path
from tqdm import tqdm
from src.data import DatasetCatalog
from src.data.utils import mmap_
from functools import partial
from .classes.imagenet import ID2NAME
from .imagenet import get_name_index_id
import logging
import PIL.Image as Image
import random

from .dataset import Dataset
logger = logging.getLogger(__name__)


class ImageNetC(Dataset):

    def __init__(self, dirname, split):

        self.thing_classes = [v for v in ID2NAME.values()]
        self.contains_boxes = False
        self.dirname = dirname
        self.split = split
        self.is_mapping_trivial = True

    def load_instances(self, cfg, **kwargs):
        """
        Load Pascal VOC detection annotations to Detectron2 format.

        Args:
            dirname: Contain "Annotations", "ImageSets", "JPEGImages"
            split (str): one of "train", "test", "val", "trainval"
            CLASSES: list or tuple of class names
        """
        max_size = int(cfg.DATASETS.MAX_DATASET_SIZE)
        p = Path(self.dirname)

        logger.info("Loading ImageNet-C {}".format(self.split))

        image_paths = list(p.glob(f'**/{self.split}/**/*.JPEG'))
        random.shuffle(image_paths)

        process_partial = partial(self.process_record)
        logger.info("{} image files found".format(len(image_paths)))
        dicts = []
        for record in mmap_(process_partial, tqdm(image_paths[:max_size])):
            if record is not None:
                dicts.append(record)
        return dicts

    def process_record(self, record: Path):
        with Image.open(record) as image:
            width, height = image.size
        instances = []
        category_id = record.parts[-2]  # if no annotation, it means it has to be a training image (all validation images have an annotation file)
        severity = int(record.parts[-3])  # if no annotation, it means it has to be a training image (all validation images have an annotation file)
        corruption = record.parts[-4]

        cat_name, cat_index, cat_id = get_name_index_id(category_id)
        instances.append({"category_id": cat_id, "category_index": cat_index, "supercategory_index": cat_index,
                          "supercategory_name": cat_name})

        r = {
            "file_name": record,
            "height": height,
            "width": width,
            "corruption": corruption,
            "severity": severity,
            "annotations": instances}
        return r
        

def register_imagenet_c(name, dirname, split):

    dataset = ImageNetC(dirname, split)
    DatasetCatalog.register(name, dataset)


if __name__ == "__main__":
    root = os.getenv("DATASET_DIR")
    imagenet_c_root = ospjoin(root, "imagenet_c")
    os.makedirs(imagenet_c_root, exist_ok=True)
    print("Which action to do ? (enter corresponding digit) \n")

    print("1 - Download \n")
    print("2 - Prepare split  \n")

    action = int(input())

    if action == 1:
        for key in tqdm(["blur", "digital", "extra", "noise", "weather"]):
            url = f"https://zenodo.org/record/2235448/files/{key}.tar?download=1"
            response = requests.get(url, stream=True)
            file = tarfile.open(fileobj=response.raw, mode="r|gz")
            file.extractall(path=imagenet_c_root)

    elif action == 2:

        val_dest = ospjoin(imagenet_c_root, 'val')
        test_dest = ospjoin(imagenet_c_root, 'test')

        os.makedirs(val_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)

        val_corruptions = ["brightness", "contrast", "defocus_blur", 
                           "elastic_transform", "fog", "frost", "gaussian_blur",
                           "gaussian_noise", "zoom_blur"]

        test_corruptions = ["glass_blur", "impulse_noise", "jpeg_compression", "motion_blur",
                            "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise"]

        for dest, all_corruptions in zip([val_dest, test_dest], [val_corruptions, test_corruptions]):
            for cor_name in tqdm(all_corruptions):

                source_dir = ospjoin(imagenet_c_root, cor_name)
                target_dir = ospjoin(dest, cor_name)
                os.makedirs(target_dir, exist_ok=True)

                file_names = os.listdir(source_dir)
                for file_name in file_names:
                    shutil.move(os.path.join(source_dir, file_name), target_dir)

