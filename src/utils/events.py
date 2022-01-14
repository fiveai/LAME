
import datetime
import json
import logging
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from contextlib import contextmanager
import torch
from fvcore.common.history_buffer import HistoryBuffer
from typing import List, Tuple, Dict, Union, Optional
from src.utils.file_io import PathManager
from src.structures import ImageList
from os.path import join as ospjoin
import numpy as np
__all__ = [
    "get_event_storage",
    "JSONWriter",
    "TensorboardXWriter",
    "EventStorage",
    "NumpyWriter",
]

_CURRENT_STORAGE_STACK = []


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


class EventWriter:
    """
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    """

    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class JSONWriter(EventWriter):
    """
    Write scalars to a json file.

    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:
    ::
        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 19,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 39,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...

    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._file_handle = PathManager.open(json_file, "a")
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        to_save = defaultdict(dict)

        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            # keep scalars that have not been written
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter["iteration"] = itr
            self._file_handle.write(json.dumps(scalars_per_iter, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self._file_handle.close()


class TensorboardXWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, max_visu_frames: int = 50, window_size: int = 20, save_plots_too: bool = False, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir, **kwargs)
        self._last_write = -1
        self._save_plots_too = save_plots_too
        self._max_visu_frames = max_visu_frames
        if save_plots_too:
            plot_dir = os.path.join(log_dir, 'saved_figures')
            os.makedirs(plot_dir, exist_ok=True)
            self._plot_dir = plot_dir

    def write(self):
        storage = get_event_storage()
        global_iter = storage.iter
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                self._writer.add_scalar(k, v, iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self._writer.add_image(img_name, img, step_num)
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                params["global_step"] = global_iter
                self._writer.add_histogram(**params)
            storage.clear_histograms()

        if len(storage._figures) >= 1:
            for fig_params in storage._figures:
                fig_params["global_step"] = global_iter
                self._writer.add_figure(**fig_params)
                if self._save_plots_too:
                    self.save_figure(fig_params)
                plt.close(fig_params["figure"])
            storage.clear_figures()

        if len(storage._video_data) >= 1:
            video_dic = storage._video_data
            for video_id in video_dic:
                chronological_indexes = np.argsort(video_dic[video_id]["timestep"])
                sorted_videos = [video_dic[video_id]["video"][i] for i in chronological_indexes]
                full_video = ImageList.from_tensors(sorted_videos[:self._max_visu_frames])  # [T, C, H, W]
                full_video = np.array(full_video.tensor)  # [T, C, H, W]
                self._writer.add_video(video_id, np.expand_dims(full_video, 0))

                # Possibly saving some frames to .png
                if self._save_plots_too:
                    for frame_id in range(min(30, len(full_video))):
                        frame = np.transpose(full_video[frame_id], (1, 2, 0))  # [H, W, C]
                        cut_w, cut_h = (frame[0, :, 0] != 0).sum(), (frame[:, 0, 0] != 0).sum()
                        if cut_h != 0 and cut_w != 0:
                            plt.imshow(frame[:cut_h, :cut_w])
                        else:
                            plt.imshow(frame)
                        root = ospjoin(self._plot_dir, "frames", video_id)
                        os.makedirs(root, exist_ok=True)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(ospjoin(root, f"{frame_id}.png"))
                        plt.clf()

            storage.clear_videos()

    def save_figure(self, figure_dic):
        fig = figure_dic["figure"]
        root = ospjoin(self._plot_dir, figure_dic["tag"])
        os.makedirs(root, exist_ok=True)
        name = ospjoin(root, "{}.png".format(figure_dic["global_step"]))
        fig.savefig(name, bbox_inches="tight")

    def close(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()


class NumpyWriter(EventWriter):
    """
    Write all scalars to numpy files. That can be used later to make camera-ready plots
    """
    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        self._last_write = -1
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)

    def pad_array(self, array, target_size, axis, pad_value=255):
        if array.shape[axis] == target_size:
            return array
        else:
            shape = array.shape
            pad_shape = list(shape)
            pad_shape[axis] = target_size - shape[axis]
            pad_array = pad_value * np.ones(pad_shape)
            return np.concatenate([array, pad_array], axis=axis)

    def write(self):
        storage = get_event_storage()

        # --------- Scalars ------------
        # ------------------------------
        for k, history_buffer in storage._history.items():
            path = os.path.join(self.log_dir, f"{k}.npy")
            values = [t[0] for t in history_buffer.values()]
            timesteps = [t[1] for t in history_buffer.values()]
            array = np.array([timesteps, values])
            np.save(path, array)

        # --------- Arrays ------------
        # -----------------------------
        for k, array_list in storage._arrays.items():
            path = os.path.join(self.log_dir, f"{k}.npy")        

            # ---- If arrays can be concatenated, they will be. Otherwise, will saved as pickle objects ---
            array = np.array(array_list)
            np.save(path, array)

    def close(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()


class EventStorage:
    """
    The user-facing class that provides metric storage functionalities.

    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ""
        self._vis_data = []
        self._histograms = []
        self._figures = []
        self._video_data = defaultdict(lambda: defaultdict(list))
        self._arrays = defaultdict(list)

    def ingest_storage(self, storage):
        """
        Allows to incorporate the metrics from another storage. Useful e.g for video adaptation. Each
        video has its own "inner" storage, and at the end of each video, we want to merge the inner
        storage with a global storage that keeps track of important metrics across all videos.
        """

        # -------- Ingest scalars --------
        # --------------------------------
        for k, (v, iter) in storage.latest().items():
            self.put_scalar(k, v, False)

        # -------- Ingest visual data --------
        # ------------------------------------
        if len(storage._vis_data) >= 1:
            for img_name, img, _ in storage._vis_data:
                self.put_image(img_name, img)

        if len(storage._video_data) >= 1:
            video_dic = storage._video_data
            self._video_data = {**self._video_data, **video_dic}

        # -------- Ingest histograms --------
        # ------------------------------------
        if len(storage._histograms) >= 1:
            self._histograms += storage._histograms

        # -------- Ingest arrays -------------
        # ------------------------------------
        if len(storage._arrays) >= 1:
            for k, array in storage._arrays.items():
                self._arrays[k] += array

    def put_image(self, img_name: str, img_tensor: Union[np.ndarray, torch.Tensor]):
        """
        Add an `img_tensor` associated with `img_name`, to be shown on
        tensorboard.

        Args:
            img_name (str): The name of the image to put into tensorboard.
            img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
                Tensor of shape `[channel, height, width]` where `channel` is
                3. The image format should be RGB. The elements in img_tensor
                can either have values in [0, 1] (float32) or [0, 255] (uint8).
                The `img_tensor` will be visualized in tensorboard.
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def put_video(self, video_name, frame_list, frame_ids_list):
        """
        Add an `img_tensor` associated with `img_name`, to be shown on
        tensorboard.

        Args:
            video_name (str): The name of the image to put into tensorboard.
            video_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
                Tensor of shape `[time, channel, height, width]` where `channel` is
                3. The image format should be RGB. The elements in img_tensor
                can either have values in [0, 1] (float32) or [0, 255] (uint8).
                The `img_tensor` will be visualized in tensorboard.
        """
        self._video_data[video_name]["video"] += frame_list
        self._video_data[video_name]["timestep"] += frame_ids_list

    def put_scalar(self, name, value, smoothing_hint=True):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.

                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        name = self._current_prefix + name
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = (value, self._iter)

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (
                existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        """
        Put multiple scalars from keyword arguments.

        Examples:

            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)

    def put_array(self, name, value):
        """
        Put multiple scalars from keyword arguments.

        Examples:

            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        self._arrays[name].append(value)

    def put_histogram(self, hist_name, hist_tensor, bins='auto'):
        """
        Create a histogram from a tensor.

        Args:
            hist_name (str): The name of the histogram to put into tensorboard.
            hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
                into a histogram.
            bins (int): Number of histogram bins.
        """

        # Create a histogram with PyTorch
        # hist_counts = torch.histc(hist_tensor.float(), bins=bins)
        # hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)

        # Parameter for the add_histogram_raw function of SummaryWriter
        hist_params = dict(
            tag=hist_name,
            values=hist_tensor,
            bins=bins,
            global_step=self._iter,
        )
        self._histograms.append(hist_params)

    def put_figures(self, fig_list):
        """
        """
        self._figures.extend(fig_list)

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20) -> Dict[str, Tuple[float, int]]:
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.

        This provides a default behavior that other writers can use.
        """
        result = {}
        for k, (v, itr) in self._latest_scalars.items():
            result[k] = (
                self._history[k].median(window_size) if self._smoothing_hints[k] else v,
                itr,
            )
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should either: (1) Call this function to increment storage.iter when needed. Or
        (2) Set `storage.iter` to the correct iteration number before each iteration.

        The storage will then be able to associate the new data with an iteration number.
        """
        self._iter += 1

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)

    @property
    def iteration(self):
        # for backward compatibility
        return self._iter

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix

    def clear_images(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._vis_data = []

    def clear_arrays(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._arrays = defaultdict(list)

    def clear_videos(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._video_data = defaultdict(lambda: defaultdict(list))

    def clear_histograms(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []

    def clear_figures(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._figures = []
