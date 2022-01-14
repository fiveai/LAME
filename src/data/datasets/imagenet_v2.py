import requests
import tarfile
import os
from pathlib import Path
import PIL.Image as Image
from tqdm import tqdm
import logging
from functools import partial
from src.data import DatasetCatalog
from src.data.utils import mmap_
import random
from .classes.imagenet import ID2NAME, INDEX2ID
from .dataset import Dataset
logger = logging.getLogger(__name__)


class ImageNetV2(Dataset):

    def __init__(self, dirname):

        self.thing_classes = [v for v in ID2NAME.values()]
        self.contains_boxes = False
        self.dirname = dirname
        self.is_mapping_trivial = True

    def load_instances(self, cfg, **kwargs):

        # ----------- Grab image paths -------------

        subset = 'imagenetv2-matched-frequency-format-val'
        max_size = int(cfg.DATASETS.MAX_DATASET_SIZE)
        logger.info("Loading imagenet {}".format(subset))
        p = Path(self.dirname)
        image_paths = list(p.glob(f'**/{subset}/**/*.jpeg'))
        random.shuffle(image_paths)

        process_partial = partial(self.process_record)
        dicts = []
        for record in mmap_(process_partial, tqdm(image_paths[:max_size])):
            if record is not None:
                dicts.append(record)
        return dicts

    def generate_mappings(self, cfg):   

        trivial_mapping = {i: i for i in range(len(self.thing_classes))}
        return trivial_mapping, trivial_mapping

    def process_record(self, record: Path):
        with Image.open(record) as image:
            width, height = image.size
        instances = []
        cat_index = int(record.parts[-2])  # if no annotation, it means it has to be a training image (all validation images have an annotation file)
        cat_id = INDEX2ID[cat_index]
        cat_name = ID2NAME[cat_id]

        instances.append({"category_id": cat_id, "category_index": cat_index, "category_name": cat_name,
                          "supercategory_id": cat_id, "supercategory_index": cat_index, "supercategory_name": cat_name})

        r = {
            "file_name": record,
            "height": height,
            "width": width,
            "annotations": instances}
        return r


def register_imagenet_v2(name, dirname):

    dataset = ImageNetV2(dirname)
    DatasetCatalog.register(name, dataset)


if __name__ == "__main__":

    root = os.getenv("DATASET_DIR")
    imagenet_v2_root = os.path.join(root, "imagenet_v2")
    os.makedirs(imagenet_v2_root, exist_ok=True)
    print("Which action to do ? (enter corresponding digit) \n")

    print("1 - Download \n")

    action = int(input())

    if action == 1:
        for key in tqdm(["matched-frequency", "threshold0.7", "top-images"]):
            url = f"https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-{key}.tar.gz"
            response = requests.get(url, stream=True)
            file = tarfile.open(fileobj=response.raw, mode="r|*")
            file.extractall(path=imagenet_v2_root)