import numpy as np
import os
import json
import requests
import zipfile
import io
from os.path import join as ospjoin
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from src.data import DatasetCatalog
from src.data.utils import mmap_
from functools import partial
from collections import defaultdict
import random
from .classes.imagenet import ID2INDEX
from .class_mapper import build_mapper
from .dataset import Dataset

# SUBSETS = ["LaSOT", "ArgoVerse", "BDD"]
# SUBSETS = ["ArgoVerse"]


class TAO(Dataset):

    def __init__(self, dirname, split):

        self.selected_datasets = ["LaSOT"]
        relevant_categories = find_relevant_categories(dirname, split, self.selected_datasets)
        self.thing_classes = [process_synset(x['synset']).split('.')[0] for x in relevant_categories]
        self.synsets = [process_synset(x["synset"]) for x in relevant_categories]
        self.cat_ids = [x["id"] for x in relevant_categories]
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
                                                target_synsets=self.synsets)

        self.thing_classes = mapping_res["new_names"]
        self.cat_ids = mapping_res["new_ids"]
        self.synsets = mapping_res["new_synsets"]
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
        """
        """

        # === Load metadata/paths in memory ===

        p = Path(self.dirname)
        with open(p / f"annotations/{self.split}.json", "rb") as f:
            main_dic = json.load(f)

        annotation_ids = np.array([x["image_id"] for x in main_dic['annotations']])
        images = main_dic["images"]
        annotations = np.array(main_dic["annotations"])
        random.shuffle(images) 

        # === Load records in memory ===

        if not hasattr(self, "subid2super"):
            self.filtered_unmapped_classes(cfg)

        process_fn = partial(self.process_record,
                             dirname=self.dirname,
                             annotation_ids=annotation_ids,
                             images=images,
                             annotations=annotations,
                             keep_multi_object=cfg.DATASETS.MULTI_OBJECT)
        dicts = []
        max_size = int(cfg.DATASETS.MAX_DATASET_SIZE)
        for record in mmap_(process_fn, tqdm(range(len(images[:max_size])))):
            if record is not None:
                dicts.append(record)

        return dicts

    def process_record(self,
                       index: int,
                       dirname: str,
                       annotations: List[Dict],
                       annotation_ids: List[int],
                       images: List[Dict],
                       keep_multi_object: bool) -> Dict[str, Any]:

        image_dic = images[index]
        # {'id': 0, 'video': 'train/YFCC100M/v_f69ebe5b731d3e87c1a3992ee39c3b7e',
        #  '_scale_task_id': '5de800eddb2c18001a56aa11', 'width': 640, 'height': 480,
        #  'file_name': 'train/YFCC100M/v_f69ebe5b731d3e87c1a3992ee39c3b7e/frame0391.jpg',
        #  'frame_index': 390, 'license': 0, 'video_id': 0}
            
        dataset = image_dic["file_name"].split('/')[1]

        #  ---- Only keep frames that belong to the subset ----

        if dataset in self.selected_datasets:
            image_dic["file_name"] = ospjoin(dirname, "frames", image_dic["file_name"])
            all_objects = annotations[annotation_ids == image_dic["id"]]
            
            instances = []
            for obj in all_objects:
                trackid = obj['track_id']
                frame_id = obj['image_id']
                #  ---- Only keep objects whose category can be mapped ----

                category_id = obj['category_id']
                if category_id in self.cat_ids:
                    category_index = self.cat_ids.index(category_id)
                    category_name = self.thing_classes[category_index]
                    category_synset = self.synsets[category_index]
                    instances.append(
                        {"supercategory_index": category_index,
                         "supercategory_id": category_id,
                         "frame_id": frame_id,
                         "supercategory_name": category_name,
                         "track": trackid,
                         "synset": category_synset}
                    )
            is_multi_object = (len(np.unique([x["supercategory_index"] for x in instances])) > 1)
            if not keep_multi_object and is_multi_object:
                return None
            elif not len(instances):
                return None
            else:
                image_dic["annotations"] = instances
                image_dic["track"] = 0
                image_dic["frame_id"] = image_dic["id"]
                return image_dic
        else:
            return None


def register_tao(name, dirname, split):

    dataset = TAO(dirname, split)
    DatasetCatalog.register(name, dataset)


def find_relevant_categories(dirname, split, selected_datasets) -> List[Dict]:
    """
    Only selects the categories that are relevant to the current selection of datasets. Each category is represented
    as a dictionnary that contains all important infos.
    """
    p = Path(dirname)
    if not os.path.exists(p / f"annotations/{split}.json"):
        return []
    with open(p / f"annotations/{split}.json", "rb") as f:
        main_dic = json.load(f)
    video2dataset = {x["id"]: x["metadata"]["dataset"] for x in main_dic["videos"]}
    classes = list(set([x["category_id"] for x in main_dic["annotations"] \
                        if video2dataset[x["video_id"]] in selected_datasets
                        ])
                   )
    relevant_categories = [cat for cat in main_dic["categories"] if cat["id"] in classes]
    return relevant_categories


def process_synset(synset: str):
    if "stop_sign" in synset:
        return 'street_sign.n.01'
    else:
        return synset


if __name__ == "__main__":

    import shutil
    N = 10
    root = os.getenv("DATASET_DIR")
    tao_root = ospjoin(root, "TAO")

    ## Check some samples
    print("Which action to do ? (enter corresponding digit) \n")

    print("0 - Download \n")
    print("1 - Visualize samples \n")
    print("2 - Create trainval split \n")

    action = int(input())

    if action == 0:
        print("Downloading data ... \n")
        for url in tqdm(["https://motchallenge.net/data/1-TAO_TRAIN.zip",
                         "https://motchallenge.net/data/2-TAO_VAL.zip"]):
            response = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(tao_root)

    # ============ Sample visualization ============
    if action == 1:
        split = "train"
        print("Please enter the synset of interest (e.g person.n.01) \n")
        dataset = TAO(root, split)
        synset_of_interest = str(input())
        with open(ospjoin(tao_root, "annotations", f"{split}.json"), "rb") as f:
            main_dic = json.load(f)
        relevant_cat = find_relevant_categories(tao_root, split, dataset.selected_datasets)
        catid2synset = {x["id"]: x["synset"] for x in relevant_cat}
        synset2def = {x["synset"]: x["def"] for x in relevant_cat}
        video2dataset = {x["id"]: x["metadata"]["dataset"] for x in main_dic["videos"]}
        relevant_image_ids = []
        for obj in main_dic["annotations"]:
            if video2dataset[obj["video_id"]] in dataset.selected_datasets and catid2synset[obj["category_id"]] == synset_of_interest:
                relevant_image_ids.append(obj["image_id"])
            if len(relevant_image_ids) > N:
                break
        relevant_names = [ospjoin(tao_root, "frames", x["file_name"]) for x in main_dic["images"] if x["id"] in relevant_image_ids]
        sample_dir = ospjoin("samples", synset_of_interest)
        os.makedirs(sample_dir, exist_ok=True)
        for i, path in enumerate(relevant_names):
            shutil.copyfile(path, ospjoin(sample_dir, f"{i}.jpg"))
        print("Some samples have been saved")
        print(synset2def[synset_of_interest])

    # ============ Merging train/val ============
    elif action == 2:
        with open(ospjoin(tao_root, "annotations", f"train.json"), "rb") as f:
            train_dic = json.load(f)
        with open(ospjoin(tao_root, "annotations", f"validation.json"), "rb") as f:
            val_dic = json.load(f)
        trainval_dic = {}
        for key in ["videos", "annotations", "tracks", "images", "categories"]:
            # Merge by id
            merged_values = {x["id"]: x for x in train_dic[key] + val_dic[key]}
            trainval_dic[key] = list(merged_values.values())

        with open(ospjoin(tao_root, "annotations", f"trainval.json"), "w") as f:
            json.dump(trainval_dic, f)
        print("Dictionnary succesfully saved at {}".format(ospjoin(tao_root, "annotations", f"trainval.json")))
    else:
        print("Entered digit does not correspond to any action.")
