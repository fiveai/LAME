import logging
from collections import UserDict
from typing import List

from src.utils.logger import log_first_n

__all__ = ["DatasetCatalog"]


class _DatasetCatalog(UserDict):
    """
    A global dictionary that stores information about the datasets and how to obtain them.

    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "imagenet_val")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.

    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionalities in `data/build.py,data/detection_transform.py`.

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """

    def register(self, name, dataset):
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
                It must return the same results if called multiple times.
        """
        # for fn in fn_dict.values():
        #     assert callable(fn), "You must register a function with `DatasetCatalog.register`!"
        if name in self:
            log_first_n(
                logging.WARNING,
                "Dataset '{}' is already registered!".format(name),
                n=10,
            )
        else:
            self[name] = dataset

    def get(self, data_name):
        """
        Call the registered function and return its results.

        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".

        Returns:
            list[dict]: dataset annotations.
        """
        try:
            f = self[data_name]
        except KeyError as e:
            raise KeyError(
                "Dataset '{}' is not registered! Available datasets are: {}".format(
                    data_name, ", ".join(list(self.keys()))
                )
            ) from e
        return f

    def list(self) -> List[str]:
        """
        List all registered datasets.

        Returns:
            list[str]
        """
        return list(self.keys())

    def remove(self, name):
        """
        Alias of ``pop``.
        """
        self.pop(name)

    def __str__(self):
        return "DatasetCatalog(registered datasets: {})".format(", ".join(self.keys()))

    __repr__ = __str__


DatasetCatalog = _DatasetCatalog()
DatasetCatalog.__doc__ = (
    _DatasetCatalog.__doc__
    + """
    .. automethod:: src.data.catalog.DatasetCatalog.register
    .. automethod:: src.data.catalog.DatasetCatalog.get
"""
)