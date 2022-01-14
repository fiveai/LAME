# -**-


from pathlib import Path
from typing import Dict
from tqdm import tqdm
import random
import logging
import os

from src.data import DatasetCatalog
from functools import partial
from src.data.utils import mmap_
from collections import defaultdict
import PIL.Image as Image

from .dataset import Dataset
from .classes.imagenet import \
        (ID2INDEX,
         ID2NAME,
         INDEX2ID)
from .imagenet_helpers import common_superclass_wnid, ImageNetHierarchy

logger = logging.getLogger(__name__)


class ImageNetC_16(Dataset):

    def __init__(self, dirname, split):

        self.contains_boxes = False
        self.dirname = dirname
        self.split = split
        self.is_mapping_trivial = False
        self.subset_name = 'geirhos_16'

        # === Obtain id2super mapping ===

        metadata_path = Path('.') / 'src' / 'data' / 'datasets' / 'class_mapper'
        imagenet_path = os.path.join(*(Path(self.dirname).parts[:-1]), 'ilsvrc12')
        in_hier = ImageNetHierarchy(imagenet_path, metadata_path)
        superclass_wnid = common_superclass_wnid(self.subset_name)
        class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

        # superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(50,
        #                                                                     balanced=False)

        id2super = defaultdict(lambda: {})
        for index, (class_set, super_id_, super_name) in enumerate(zip(class_ranges, superclass_wnid, label_map.values())):
            relevant_class_ids = [INDEX2ID[x] for x in class_set]
            for class_id in relevant_class_ids:
                assert class_id not in id2super, f"Class {class_id} seems to be a subclass of both {super_name} and {id2super[class_id]['super_name']}"
                id2super[class_id]["super_id"] = super_id_
                id2super[class_id]["super_index"] = index
                id2super[class_id]["super_name"] = super_name

        self.id2super = dict(id2super)
        self.thing_classes = list(label_map.values())
        self.cat_ids = superclass_wnid
        self.synsets = superclass_wnid

    def load_instances(self, cfg, **kwargs):

        max_size = int(cfg.DATASETS.MAX_DATASET_SIZE)
        logger.info("Loading ImageNet-C-16 {} with {} superclasses and {} subclasses".format(self.split, len(self.thing_classes), len(self.id2super)))

        image_paths = []
        p = Path(self.dirname)

        # Recover relevant images
        image_paths = list(p.glob(f'**/{self.split}/**/*.JPEG'))
        image_paths = [x for x in image_paths if x.parts[-2] in self.id2super]
        random.shuffle(image_paths)
        logger.info("{} image files found".format(len(image_paths)))

        # Get records
        dicts = []
        process_partial = partial(self.process_record, id2super=dict(self.id2super))
        for record in mmap_(process_partial, tqdm(image_paths[:max_size])):
            if record is not None:
                dicts.append(record)

        return dicts

    def filtered_unmapped_classes(self, cfg):
        pass

    def generate_mappings(self, cfg):

        sub2super = {ID2INDEX[k]: v['super_index'] for k, v in self.id2super.items()}
        super2sub = defaultdict(list)
        for sub, super_ in sub2super.items():
            super2sub[super_].append(sub)

        return sub2super, super2sub

    def process_record(self, jpeg_file: Path, id2super: Dict):
        r = {
            "file_name": jpeg_file,
        }
        with Image.open(jpeg_file) as image:
            width, height = image.size
            r["width"] = width
            r["height"] = height
        instances = []
        cat_id = jpeg_file.parts[-2]  # if no annotation, it means it has to be a training image (all validation images have an annotation file)
        if cat_id in id2super:
            instances.append({"category_id": cat_id,
                              "category_index": ID2INDEX[cat_id],
                              "category_name": ID2NAME[cat_id],
                              "supercategory_id": id2super[cat_id]['super_id'],
                              "supercategory_index": id2super[cat_id]['super_index'],
                              "supercategory_name": id2super[cat_id]['super_name']}
                               )  # noqa: E124
            r["annotations"] = instances
            return r
        else:
            return None


def register_imagenet_c_16(name, dirname, split):

    dataset = ImageNetC_16(dirname, split)
    DatasetCatalog.register(name, dataset)