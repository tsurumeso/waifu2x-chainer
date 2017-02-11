import six
import numpy as np
import multiprocessing

from lib import iproc
from lib.pairwise_transform import pairwise_transform


class DatasetSampler():

    def __init__(self, datalist, config, repeat=False):
        self.datalist = datalist
        self.config = config
        self.repeat = repeat
        self.worker = None
        self.data_queue = None
        self.finalized = None
        self.dataset = None
        self.running = False
        self._switch = True
        self._init_process()

    def __del__(self):
        self.finalize()

    def finalize(self):
        if self.running:
            self.finalized.set()
            garbage = self.data_queue.get()
            self.worker.join()
            del garbage

    def reload_switch(self):
        self._switch = True

    def _init_process(self):
        self.data_queue = multiprocessing.Queue()
        self.finalized = multiprocessing.Event()
        args = [self.datalist, self.data_queue, self.config, self.finalized]
        self.worker = multiprocessing.Process(target=_worker, args=args)
        self.worker.start()
        self.running = True

    def get(self):
        if self.running and self._switch:
            self.dataset = self.data_queue.get()
            self.worker.join()
            self.running = False
            self._switch = False
            if self.repeat:
                self._init_process()
        return self.dataset


def _worker(datalist, out_queue, cfg, finalized):
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
    out_queue.put([x, y])
    del x, y
