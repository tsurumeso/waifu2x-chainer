import os
import six
import numpy as np
import multiprocessing
from tempfile import NamedTemporaryFile

from lib import iproc
from lib.pairwise_transform import pairwise_transform


class DatasetSampler():

    def __init__(self, datalist, config):
        self.datalist = datalist
        self.config = config

        self.worker = None
        self.dataset = None
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
        args = [self.datalist, self.config, self._queue, self._finalized]
        self.worker = multiprocessing.Process(target=_worker, args=args)
        self.worker.daemon = True
        self.worker.start()
        self._running = True

    def get(self):
        if self._reload:
            cache_name = self._queue.get()
            self.worker.join()
            with np.load(cache_name) as cached_arr:
                self.dataset = cached_arr['x'], cached_arr['y']
            os.remove(cache_name)
            self._running = False
            if self._init:
                self._init_process()
            self._reload = False
        return self.dataset

    def save_images(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if self.dataset is None:
            x, y = self.get()
        else:
            x, y = self.dataset

        digits = int(np.log10(len(x) * 2)) + 1
        for i, (ix, iy) in enumerate(zip(x, y)):
            ix = iproc.to_image(ix, self.config.ch, False)
            iy = iproc.to_image(iy, self.config.ch, False)
            header = 'image_%s' % str(i).zfill(digits)
            ix.save(os.path.join(dir, header + '_x.png'))
            iy.save(os.path.join(dir, header + '_y.png'))


def _worker(datalist, cfg, queue, finalized):
    sample_size = cfg.patches * len(datalist)
    x = np.zeros(
        (sample_size, cfg.ch, cfg.insize, cfg.insize), dtype=np.uint8)
    y = np.zeros(
        (sample_size, cfg.ch, cfg.crop_size, cfg.crop_size), dtype=np.uint8)

    for i in six.moves.range(len(datalist)):
        if finalized.is_set():
            break
        img = iproc.read_image_rgb_uint8(datalist[i])
        xc_batch, yc_batch = pairwise_transform(img, cfg)
        x[cfg.patches * i:cfg.patches * (i + 1)] = xc_batch[:]
        y[cfg.patches * i:cfg.patches * (i + 1)] = yc_batch[:]

    with NamedTemporaryFile(delete=False) as cache:
        np.savez(cache, x=x, y=y)
        queue.put(cache.name)
        del x, y
