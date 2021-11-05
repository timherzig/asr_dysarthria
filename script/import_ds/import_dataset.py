from .import_hu import import_hu
from .import_ua import import_ua
from .import_torgo import import_torgo

# returns a list of datasets, if len == 1, then speaker independent, if len > 1 then speaker dependent
def import_dataset(name, local, test_train, t='train'):
    # manually change paths

    if not test_train:
        ds = []
        if name.lower() == 'torgo':
            if local:
                ds = import_torgo('/home/tim/Documents/Datasets/torgo/TORGO', False, t)
            else:
                ds = import_torgo('/work/herzig/datasets/torgo/TORGO', False, t)
        elif name.lower() == 'hu':
            if local:
                ds = import_hu('/home/tim/Documents/Datasets/hu_final', False)
            else:
                ds = import_hu('/work/herzig/datasets/hu_final', False)
        elif name.lower() == 'uaspeech' or name.lower() == 'ua':
            if local:
                ds = import_ua('/home/tim/Documents/Datasets/uaspeech', False, t)
            else:
                ds = import_ua('/work/herzig/datasets/uaspeech', False, t)
        return ds

    if test_train:
        if name.lower() == 'torgo':
            if local:
                return import_torgo('/home/tim/Documents/Datasets/torgo/TORGO', True, t)
            else:
                return import_torgo('/work/herzig/datasets/torgo/TORGO', True, t)
        elif name.lower() == 'hu':
            if local:
                return import_hu('/home/tim/Documents/Datasets/hu_final', True)
            else:
                return import_hu('/work/herzig/datasets/hu_final', True)
        elif name.lower() == 'uaspeech' or name.lower() == 'ua':
            if local:
                return import_ua('/home/tim/Documents/Datasets/uaspeech', True, t)
            else:
                return import_ua('/work/herzig/datasets/uaspeech', True, t)
