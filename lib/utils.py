import os
import sys
import chainer
import chainer.functions as F
import chainer.links as L


def get_model_module(model):
    if isinstance(model, chainer.Chain):
        child = model.children().next()
        return child.xp


def load_datalist(dir):
    files = os.listdir(dir)
    datalist = []
    for file in files:
        datalist.append(os.path.join(dir, file))
    return datalist


def offset_size(model):
    offset = 0
    if hasattr(model, 'offset'):
        offset = model.offset
    else:
        for child in model.children():
            if isinstance(child, chainer.Link):
                offset += child.W.data.shape[2] - 1
    return offset


def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)
        if isinstance(child, chainer.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print 'Copy %s' % child.name
