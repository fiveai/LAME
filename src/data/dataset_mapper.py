
import copy
import logging
from typing import List, Union, Dict, Any

from src.config import configurable
from src.utils.logger import log_every_n_seconds
from . import utils
from . import transforms as T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        # fmt: on
        # logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        log_every_n_seconds(logging.INFO,
                            f"[DatasetMapper] Augmentations used in {mode}: {augmentations}",
                            n=3600,
                            name=__name__)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, record: Dict[str, Any]):
        """
        Processes a record.
        Args:
            record (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in src accept
        """
        record = copy.deepcopy(record)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(record["file_name"], format=self.image_format)
        # utils.check_image_size(record, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in record:
            sem_seg_gt = utils.read_image(record.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        crop = record["crop"] if "crop" in record else None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt, crop=crop)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        record["image"] = image
        # record["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in record:
            # USER: Modify this if you want to keep them for some reason.

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in record.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape
            )

            record["instances"] = instances
            # record["instances"] = utils.filter_empty_instances(instances)
        # assert len(record["instances"]), record
        return record

