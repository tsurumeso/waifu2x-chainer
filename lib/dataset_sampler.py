import os
import six
import numpy as np
import multiprocessing
from tempfile import NamedTemporaryFile

from lib import iproc
from lib.pairwise_transform import pairwise_transform


class DatasetSampler():

    def __init__(self, datalist, config, repeat=False):
        self.datalist = datalist
        self.config = config
        self.repeat = repeat
        self.worker = None
        self.name_queue = None
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
            garbage = self.name_queue.get(timeout=0.5)
            self.worker.join()
            os.remove(garbage)

    def reload_switch(self):
        self._switch = True

    def _init_process(self):
        self.name_queue = multiprocessing.Queue()
        self.finalized = multiprocessing.Event()
        args = [self.datalist, self.name_queue, self.config, self.finalized]
        self.worker = multiprocessing.Process(target=_worker, args=args)
        self.worker.daemon = True
        self.worker.start()
        self.running = True

    def get(self):
        if self.running and self._switch:
            cache_name = self.name_queue.get()
            self.worker.join()
            six.print_('  * loading dataset from cache...',
                       end=' ', flush=True)
            with np.load(cache_name) as cached_arr:
                self.dataset = cached_arr['x'], cached_arr['y']
            os.remove(cache_name)
            six.print_('done')

            self.running = False
            self._switch = False
            if self.repeat:
                self._init_process()

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


def _worker(datalist, name_queue, cfg, finalized):
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
        name_queue.put(cache.name)
        del x, y
