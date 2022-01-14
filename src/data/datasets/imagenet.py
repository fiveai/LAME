from pathlib import Path
from typing import List
from tqdm import tqdm
from src.data import DatasetCatalog
from .classes.imagenet import \
        (ID2INDEX,
         ID2NAME,
         NAME2ID,
         NAME2INDEX
         )
# import nltk
# nltk.download('wordnet')
from nltk.corpus.reader.wordnet import Synset
import logging
from functools import partial
import random
import PIL.Image as Image

from src.data.utils import mmap_
from .class_mapper.build_imagenet_graph import fns
from .dataset import Dataset

logger = logging.getLogger(__name__)


class ImageNet(Dataset):

    def __init__(self, dirname, split):

        self.thing_classes = [v for v in ID2NAME.values()]
        self.contains_boxes = False
        self.dirname = dirname
        self.split = split
        self.is_mapping_trivial = True

    def load_instances(self, cfg, **kwargs):

        # ----------- Grab image paths -------------

        max_size = int(cfg.DATASETS.MAX_DATASET_SIZE)
        logger.info("Loading imagenet {}".format(self.split))
        p = Path(self.dirname)
        image_paths = list(p.glob(f'**/{self.split}/**/*.JPEG'))
        logger.info("{} image files found.".format(len(image_paths)))
        random.shuffle(image_paths)

        # ----------- Process images and aggregate them -----------

        dicts = []
        process_partial = partial(self.process_record)
        for record in mmap_(process_partial, tqdm(image_paths[:max_size])):
            if record is not None:
                dicts.append(record)

        return dicts

    def generate_mappings(self, cfg):   

        trivial_mapping = {i: i for i in range(len(self.thing_classes))}
        return trivial_mapping, trivial_mapping

    def process_record(self, jpeg_file: Path):
        r = {
            "file_name": jpeg_file,
        }
        with Image.open(jpeg_file) as image:
            width, height = image.size
            r["width"] = width
            r["height"] = height
        
        instances = []
        category_id = jpeg_file.parts[-2]  # if no annotation, it means it has to be a training image (all validation images have an annotation file)
        cat_name, cat_index, cat_id = get_name_index_id(category_id)
        instances.append({"category_id": cat_id, "supercategory_id": cat_id, 
                          "supercategory_index": cat_index, "category_index": cat_index,
                          "category_name": cat_name, "supercategory_name": cat_name})
        r["annotations"] = instances
        return r
        # else:
        #     return None


def get_name_index_id(cat_id):
    try:
        cat_index = ID2INDEX[cat_id]
        cat_name = ID2NAME[cat_id]
    except:
        cat_name = cat_id
        cat_index = NAME2INDEX[cat_name]
        cat_id = NAME2ID[cat_name]
    return cat_name, cat_index, cat_id


def get_imagenet_synsets() -> List[Synset]:
    synset_list = []
    for id_ in ID2INDEX.keys():
        synset_list.append(fns["id2synset"](id_))
    return synset_list


def register_imagenet(name, dirname, split, year):

    dataset = ImageNet(dirname, split)
    DatasetCatalog.register(name, dataset)