
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2
_C.DEBUG = False
_C.OVERRIDE = False

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.PARTITION_GRANULARITY = 2
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "Classifier"
_C.MODEL.WEIGHTS = ""
_C.MODEL.NORMALIZE_INPUT = False
_C.MODEL.STANDARDIZE_INPUT = True
# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

# ---------------------------------------------------------------------------- #
# ADAPTATION options
# ---------------------------------------------------------------------------- #
_C.ADAPTATION = CN()
_C.ADAPTATION.METHOD = "NonAdaptiveMethod"
_C.ADAPTATION.VISU_PERIOD = 50
_C.ADAPTATION.ONLINE = True

_C.ADAPTATION.NUMPY_WRITER = True


# Optim params
_C.ADAPTATION.STEPS = 1
_C.ADAPTATION.LR = 0.01
_C.ADAPTATION.BETA = 0.9
_C.ADAPTATION.OPTIMIZER = "SGD"
_C.ADAPTATION.BATCH_SIZE = 16
_C.ADAPTATION.OPTIM_MOMENTUM = 0.
_C.ADAPTATION.DAMPENING = 0.
_C.ADAPTATION.WEIGHT_DECAY = 0.0001
_C.ADAPTATION.NESTEROV = True

# Visu Params
_C.ADAPTATION.MAX_FRAMES_PER_VIDEO = 1e6
_C.ADAPTATION.MAX_VISU_FRAMES = 50
_C.ADAPTATION.MAX_BATCH_PER_EPISODE = int(1e6)

# Method specific
_C.ADAPTATION.PL_THRESHOLD = 0.9

_C.ADAPTATION.LAME_KNN = 5
_C.ADAPTATION.LAME_SIGMA = 1.0
_C.ADAPTATION.LAME_AFFINITY = 'rbf'
_C.ADAPTATION.LAME_FORCE_SYMMETRY = False

_C.ADAPTATION.SHOT_BETA = 0.

_C.ADAPTATION.PARAMS2ADAPT = "all-BN"
_C.ADAPTATION.HPARAMS2TUNE = ['ADAPTATION.PARAMS2ADAPT', 'ADAPTATION.LR', 'MODEL.BACKBONE.BN_MOMENTUM', 'ADAPTATION.OPTIM_MOMENTUM']
_C.ADAPTATION.HPARAMS_VALUES = [
								["0-BN", "1-BN", "all-BN"],  # noqa: E126
								[0.001, 0.01, 0.1],  # noqa: W191
								[0., 0.1, 1.0],  # noqa: W191
								[0., 0.9],  # noqa: W191
							    ]  # noqa: E126


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.RESIZE_MODE = "shortest_edge"
_C.INPUT.MIN_SIZE = 256
_C.INPUT.MAX_SIZE = 1000

# `True` if cropping is used for data augmentation during training
_C.INPUT.CROP = CN()
_C.INPUT.CROP.ENABLED = True
_C.INPUT.CROP.SIZE = [224, 224]


_C.INPUT.FORMAT = "BGR"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.MAX_DATASET_SIZE = 5e5
_C.DATASETS.MAX_SIZE_PER_EPISODE = 5e4
_C.DATASETS.ROOT_DIR = "/data_ssd/datasets/"

_C.DATASETS.IID = True
_C.DATASETS.IMBALANCE_SHIFT = False
_C.DATASETS.IMBALANCE_TYPE = 'zipf'  # ['zipf', 'dirichlet']

_C.DATASETS.MULTI_OBJECT = False
_C.DATASETS.ADAPTATION = ["imagenet_vid"]

_C.DATASETS.MAPPER = CN()
_C.DATASETS.MAPPER.ENABLED = True
_C.DATASETS.MAPPER.THRESHOLD = 0.3  # Used for threshold mapper
_C.DATASETS.MAPPER.NAME = 'AncestralSynsetMapper'
_C.DATASETS.MAPPER.SYNSET_SIM = 'path_similarity'  # ['path_similarity']
_C.DATASETS.MAPPER.IMAGENET_GRAPH_PATH = 'src/data/datasets/class_mapper/imagenet_graph.pkl'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.BN_MOMENTUM = 0.  # 0 means not updating model's statistics. Note that ViT uses LayerNorm, which makes this argument useless.
_C.MODEL.BACKBONE.WEIGHTS = ""
# Freeze the first several stages so they are not trained.
# There are 5 stages in ResNet. The first is a convolution, and the following
# stages are each group of residual blocks.
_C.MODEL.BACKBONE.FREEZE_AT = []


# ---------------------------------------------------------------------------- #
# Classifier head options
# ---------------------------------------------------------------------------- #
_C.MODEL.CLS_HEAD = CN()
_C.MODEL.CLS_HEAD.NAME = "StandardHead"
_C.MODEL.CLS_HEAD.TYPE = "conv2d"
_C.MODEL.CLS_HEAD.DISTANCE = "dot"
_C.MODEL.CLS_HEAD.TEMPERATURE = 1.0
_C.MODEL.CLS_HEAD.IN_FEATURES = ["res5"]
_C.MODEL.CLS_HEAD.NUM_CLASSES = 1000
_C.MODEL.CLS_HEAD.POST_AGGREGATION = "max"


# ---------------------------------------------------------------------------- #
# ViT
# ---------------------------------------------------------------------------- #

_C.MODEL.VIT = CN()
_C.MODEL.VIT.NAME = ""

# ---------------------------------------------------------------------------- #
# 
# ---------------------------------------------------------------------------- #

_C.MODEL.EFFICIENT_NET = CN()
_C.MODEL.EFFICIENT_NET.NAME = ""


# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

_C.MODEL.RESNETS.DEPTH = 50
_C.MODEL.RESNETS.OUT_FEATURES = ["res5"]  # res4 for C4 backbone, res2..5 for FPN backbone

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Options: MaskedBN, FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.RESNETS.NORM = "BatchNorm2d"

# Baseline width of each group.
# Scaling this parameters will scale the width of all bottleneck layers.
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

# Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
# For R18 and R34, this needs to be set to 64
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
_C.SAVE_PLOTS = False
_C.SEED = 1
_C.CUDNN_BENCHMARK = False

