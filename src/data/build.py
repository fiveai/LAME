
import itertools
import logging
import numpy as np
import random
import torch.utils.data
from typing import List, Union, Dict, Any
from collections import defaultdict
from torch.utils.data import DataLoader
from copy import deepcopy

from src.utils.env import seed_all_rng
from src.utils.logger import _log_api_usage

    
from .common import DatasetFromList, MapDataset
from .dataset_mapper import DatasetMapper
from .catalog import DatasetCatalog
from .samplers import InferenceSampler
# from .datasets.builtin import register_all_imagenet_vid, register_all_imagenet
from src.config import CfgNode
"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_test_loader",
    "build_datasets",
    "get_dataset_dicts",
]


def get_dataset_dicts(cfg: CfgNode,
                      names: Union[str, List[str]],
                      ) -> List[Dict[str, Any]]:
    """
    Load and prepare a dataset_dict. A dataset_dict is given by a sequence of records, where each record
    is a dictionary representing an image + metadata such as its label.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    logger = logging.getLogger(__name__)
    if isinstance(names, str):
        names = [names]
    assert len(names), names

    dataset_dicts = [DatasetCatalog.get(dataset_name).load_instances(cfg) for dataset_name in names]
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))

    return dataset_dicts


def group_by_subclass(dataset_dicts: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    res = defaultdict(list)
    for record in dataset_dicts:
        res[record['annotations'][0]['category_index']].append(record)
    return res


def group_by(dataset_dicts: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    res = defaultdict(list)
    for record in dataset_dicts:
        res[record[key]].append(record)
    return res


def flatten_dict(dict_: Dict[Any, Any]) -> List[Any]:
    res = []
    for group in dict_.values():
        # if is_video:
        #     group = sorted(group, key=lambda r: r['frame_id'])
        res += group
    return res


def zipf_imbalance(num_classes: int, total_samples: int, mode: str, a=2, **kwargs) -> List[int]:
    """
    Produces the number of samples to keep for each class following a zipf distribution. We
    intentionnaly create a slight difference in the way imbalance is generated between
    validation and to avoid "overfitting" the imbalance distribution.
    """
    if mode == 'validation':
        x = 0.2  # percentage of "reigning" classes
        y = 0.8  # percentage of samples these classes are allowed to accaparate
    else:
        x = 0.1  # percentage of "reigning" classes
        y = 0.9  # percentage of samples these classes are allowed to accaparate

    valid = False
    while not valid:
        proportions = np.random.zipf(a, num_classes)
        proportions = proportions / proportions.sum()  # normalize to obtain valid proportions
        samples_per_class = (total_samples * proportions).astype(np.int32)
        sorted_samples = np.sort(samples_per_class)
        if sorted_samples[-int(x * num_classes):].sum() < int(y * total_samples):
            valid = True
    return samples_per_class


def build_datasets(cfg, dataset_names: List[str]) -> List[List[Dict[str, Any]]]:
    """
    Build n_runs random versions of the datasets.
    """

    # ------------ Build dataset  ------------
    dataset: List[Dict[str, Any]] = get_dataset_dicts(
        cfg,
        dataset_names,
    )
    _log_api_usage("dataset." + dataset_names[0])
    
    if cfg.mode == 'validation':
        n_runs = 3
    elif cfg.mode == 'benchmark':
        n_runs = 10
    elif cfg.mode == 'test':
        n_runs = 5
    else:
        n_runs = 1

    logger = logging.getLogger(__name__)

    all_datasets = []
    for run in range(n_runs):

        current_datasets = deepcopy(dataset)
        random.shuffle(current_datasets)

        # ------------ Create class imbalance ------------

        if cfg.DATASETS.IMBALANCE_SHIFT:

            logger.info(f'Adding Zipf class imbalance')
            per_class_samples = group_by_subclass(current_datasets)
            all_classes = list(per_class_samples.keys())
            samples_per_class = zipf_imbalance(num_classes=len(all_classes),
                                               mode=cfg.mode,
                                               total_samples=sum([len(x) for x in per_class_samples.values()]))
            per_class_set = {class_index: list(np.random.choice(per_class_samples[class_index], num_samples)) \
                             for class_index, num_samples in zip(all_classes, samples_per_class)}
            if cfg.DEBUG:
                logger.info(f"Detected {len(per_class_samples)} subclasses")
                logger.info(f"Samples for each subclass : {samples_per_class}")
            current_datasets = flatten_dict(per_class_set)
            random.shuffle(current_datasets)
        else:
            logger.info(f'No artificial class imbalance, creating {n_runs} with different class sequences')

        # ----------- Present the dataset as a stream of groups if non i.i.d images ------------

        is_video_dataset = ('video_id' in current_datasets[0])
        if not cfg.DATASETS.IID:
            logger.info('Non IID setting')
            if is_video_dataset:
                key = 'video_id'
                per_group_data = group_by(current_datasets, key)
            elif ("corruption" in current_datasets[0]):
                key = 'corruption'
                per_group_data = group_by(current_datasets, key)
            else:
                key = 'subclass'
                per_group_data = group_by_subclass(current_datasets)
            logger.info(f'Grouped by {key}')
            if cfg.DEBUG:
                logger.info(f'Length of 10 first groups: {[len(group) for group in per_group_data.values()][:10]}')
            current_datasets = flatten_dict(per_group_data)
        else:
            logger.info(f'IID setting, everything shuffled')

        # ----------- Cut to the maximum number of samples ------------

        current_datasets = current_datasets[:int(cfg.DATASETS.MAX_SIZE_PER_EPISODE)]
        if cfg.DEBUG:
            N = 20
            logger.info(f'First {N} class ids') 
            logger.info([r['annotations'][0]['supercategory_index'] for r in current_datasets[:N]])

        all_datasets.append(current_datasets)

    return all_datasets


def build_test_loader(cfg, dataset_dict: List[Dict[str, Any]], num_workers: int,
                      batch_size: int, serialize=True, shuffle=False) -> torch.utils.data.DataLoader:
    """
    Build the loader from a dataset_dict.
    """
    assert isinstance(dataset_dict, list)
    mapper = DatasetMapper(cfg, False)
    dataset = DatasetFromList(dataset_dict, copy=False, serialize=serialize)
    dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(size=len(dataset), shuffle=shuffle)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)
    data_loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    inputs = [x[0] for x in batch]
    indexes = [x[1] for x in batch]
    return inputs, indexes


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)
