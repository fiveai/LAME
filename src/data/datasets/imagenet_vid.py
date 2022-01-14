# -**-

import requests
import tarfile
import numpy as np
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Union
from functools import partial
from collections import defaultdict
from tqdm import tqdm
import random
import logging
from src.utils.logger import setup_logger
from src.data import DatasetCatalog
from src.data.utils import mmap_
from .classes.imagenet_vid import CLASSES
from .classes.imagenet import \
        (ID2INDEX,
         ID2NAME,
         NAME2ID,
         NAME2INDEX,
         INDEX2ID)
from .class_mapper import build_mapper
from .dataset import Dataset


class ImageNetVid(Dataset):

    def __init__(self, dirname, split):

        self.thing_classes = [t[1] for t in CLASSES.values()]
        self.cat_ids = list(CLASSES.keys())
        self.contains_boxes = True
        self.dirname = dirname
        self.split = split
        self.is_mapping_trivial = False

    def filtered_unmapped_classes(self, cfg):
        """
        Here the goal is to remove target superclasses that aren't the image of any imagenet class.
        """
        class_mapper = build_mapper(cfg)
        mapping_res = class_mapper.get_mappings(target_class_names=self.thing_classes,
                                                target_class_ids=self.cat_ids,
                                                target_synsets=None)

        self.thing_classes = mapping_res["new_names"]
        self.cat_ids = mapping_res["new_ids"]
        self.subid2super = mapping_res["imagenet2target"] 

    def generate_mappings(self, cfg):
        """
        Mostly useful for the post-hoc mapping of the logits
        """
        if not hasattr(self, "subid2super"):
            self.filtered_unmapped_classes(cfg)

        sub2super = {ID2INDEX[id_]: self.subid2super[id_]['super_index'] for id_ in self.subid2super.keys()}
        super2sub = defaultdict(list)
        for sub_index, super_index in sub2super.items():
            super2sub[super_index].append(sub_index)

        return sub2super, super2sub

    def load_instances(self, cfg):

        # === Grab and shuffle frames ===

        p = Path(self.dirname)
        max_size = int(cfg.DATASETS.MAX_DATASET_SIZE)
        image_paths = list(p.glob(f'**/{self.split}/**/*.JPEG'))
        random.shuffle(image_paths) 
        
        # === Load instances in memory ===

        dicts = []
        if not hasattr(self, "subid2super"):
            self.filtered_unmapped_classes(cfg)
        process_fn = partial(self.process_record, keep_multi_object=cfg.DATASETS.MULTI_OBJECT)
        for record in mmap_(process_fn, tqdm(image_paths[:max_size])):
            if record is not None:
                dicts.append(record)
        return dicts

    def process_record(self, jpeg_file: Path, keep_multi_object: bool):

        parts = list(jpeg_file.parts)
        parts[parts.index('Data')] = 'Annotations'
        parts[-1] = parts[-1].replace('JPEG', 'xml')
        anno_file = Path(*parts)
        
        try:
            with open(anno_file) as f:
                tree = ET.parse(f)
            r = {
                "file_name": jpeg_file,
                "frame_id": int(jpeg_file.stem),
                "video_id": jpeg_file.parts[-2],
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
                "track": 0,
            }
            instances = []
            for obj in tree.findall("object"):
                category_id = obj.find("name").text
                trackid = int(obj.find("trackid").text)
                if category_id in self.cat_ids:
                    category_index = self.cat_ids.index(category_id)
                    instances.append(
                        {"supercategory_index": category_index,
                         "supercategory_name": self.thing_classes[category_index],
                         "supercategory_id": category_id,
                         "track": trackid
                         }
                    )

            is_multi_object = (len(np.unique([x["supercategory_index"] for x in instances])) > 1)

            if not len(instances):
                return None
            elif is_multi_object and not keep_multi_object:
                return None
            else:
                r["annotations"] = instances
                return r
        except:
            print(f"Problem with file {jpeg_file}")
            return None


def register_imagenet_vid(name, dirname, split):

    dataset = ImageNetVid(dirname, split)
    DatasetCatalog.register(name, dataset)


if __name__ == "__main__":
    logger = setup_logger(name=__name__)
    root = os.getenv("DATASET_DIR")
    assert root is not None, "Please set the DATASET_DIR environment variable"
    imagenet_vid_root = os.path.join(root, "ILSVRC2015")
    os.makedirs(imagenet_vid_root, exist_ok=True)
    print("Which action to do ? (enter corresponding digit) \n")

    print("1 - Download \n")

    action = int(input())

    if action == 1:
        logger.info("Starting to download (86 GB...) may be long \n")
        # You must initialize logging, otherwise you'll not see debug output.
        requests_log = logging.getLogger("requests")
        requests_log.setLevel(logging.INFO)
        requests_log.propagate = True

        url = f"https://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz"
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=root)