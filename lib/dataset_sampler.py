import multiprocessing
import os
from tempfile import NamedTemporaryFile

import numpy as np
import six

from lib import iproc
from lib.pairwise_transform import pairwise_transform


class DatasetSampler(object):

    def __init__(self, filelist, config):
        self.filelist = filelist
        self.config = config

        self.worker = None
        self.dataset = None
        self.cache_name = None
        self._queue = None
        self._finalized = None
        self._init = False
        self._reload = True
        self._running = False

        self._init_process()

    def __del__(self):
        self.finalize()

    def finalize(self):
        if self._running:
            self._finalized.set()
            garbage = self._queue.get(timeout=0.5)
            self.worker.join()
            os.remove(garbage)

    def reload_switch(self, init=True):
        self._init = init
        self._reload = True

    def _init_process(self):
        self._queue = multiprocessing.Queue()
        self._finalized = multiprocessing.Event()
        args = [self.filelist, self.config, self._queue, self._finalized]
        self.worker = multiprocessing.Process(target=_worker, args=args)
        self.worker.daemon = True
        self.worker.start()
        self._running = True

    def wait(self):
        if self._running and self.cache_name is None:
            self.cache_name = self._queue.get()
            self.worker.join()
            self._running = False

    def get(self):
        if self._reload:
            if self._running and self.cache_name is None:
                self.cache_name = self._queue.get()
                self.worker.join()
                self._running = False
            with np.load(self.cache_name) as cached_arr:
                self.dataset = cached_arr['x'], cached_arr['y']
            os.remove(self.cache_name)
            if self._init:
                self._init_process()
            self.cache_name = None
            self._reload = False
        return self.dataset


def _worker(filelist, cfg, queue, finalized):
    sample_size = cfg.patches * len(filelist)
    x = np.zeros(
        (sample_size, cfg.ch, cfg.in_size, cfg.in_size), dtype=np.uint8)
    y = np.zeros(
        (sample_size, cfg.ch, cfg.out_size, cfg.out_size), dtype=np.uint8)

    for i in six.moves.range(len(filelist)):
        if finalized.is_set():
            break
        img = iproc.read_image_rgb_uint8(filelist[i])
        xc_batch, yc_batch = pairwise_transform(img, cfg)
        x[cfg.patches * i:cfg.patches * (i + 1)] = xc_batch[:]
        y[cfg.patches * i:cfg.patches * (i + 1)] = yc_batch[:]

    with NamedTemporaryFile(delete=False) as cache:
        np.savez(cache, x=x, y=y)
        del x, y
        queue.put(cache.name)
