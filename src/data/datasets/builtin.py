"""
This file registers pre-defined datasets.
"""

import os
from .imagenet_vid import register_imagenet_vid
from .imagenet import register_imagenet
from .imagenet_c_16 import register_imagenet_c_16
from .imagenet_c import register_imagenet_c
from .imagenet_v2 import register_imagenet_v2
from .tao import register_tao


# ==== ImageNet VID ===========
def register_all_imagenet_vid(root):
    SPLITS = [
        ("imagenet_vid_val", "ILSVRC2015", "val"),
    ]
    for name, dirname, split in SPLITS:
        register_imagenet_vid(name, os.path.join(root, dirname), split)


# ==== ImageNet-C ===========
def register_all_imagenet_c(root):
    SPLITS = [
        ("imagenet_c_val", "imagenet_c", "val"),
        ("imagenet_c_test", "imagenet_c", "test"),
    ]
    for name, dirname, split in SPLITS:
        register_imagenet_c(name, os.path.join(root, dirname), split)
        

# ==== ImageNet-C-16 ===========
def register_all_imagenet_c_16(root):
    SPLITS = [
        ("imagenet_c_16", "imagenet_c", "val"),
    ]
    for name, dirname, split in SPLITS:
        register_imagenet_c_16(name, os.path.join(root, dirname), split)


# ==== ImageNet V2 ===========
def register_all_imagenetv2(root):
    SPLITS = [
        ("imagenet_v2", "imagenet_v2"),
    ]
    for name, dirname in SPLITS:
        register_imagenet_v2(name, os.path.join(root, dirname))


# ==== ImageNet ===========
def register_all_imagenet(root):
    SPLITS = [
        ("imagenet_val", "ilsvrc12", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = "2012"
        register_imagenet(name, os.path.join(root, dirname), split, year)


# ==== TAO ===========
def register_all_tao(root):
    SPLITS = [
        ("tao_trainval", "TAO", "trainval")
    ]
    for name, dirname, split in SPLITS:
        register_tao(name, os.path.join(root, dirname), split)


if __name__.endswith(".builtin"):

    _root = os.getenv("DATASET_DIR")
    assert _root is not None, "Please set the DATASET_DIR environment variable with the following command: \
                                export DATASET_DIR=/path/to/data/dir"
    register_all_imagenet_vid(_root)
    register_all_imagenet(_root)
    register_all_imagenet_c_16(_root)
    register_all_tao(_root)
    register_all_imagenet_c(_root)
    register_all_imagenetv2(_root)
