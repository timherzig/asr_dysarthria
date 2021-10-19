from .import_hu import import_hu
from .import_ua import import_ua
from .import_torgo import import_torgo

# returns a list of datasets, if len == 1, then speaker independent, if len > 1 then speaker dependent
def import_dataset(name, local):
    ds = []
    
    # manually change paths
    
    if name.lower() == 'torgo':
        if local:
            ds = import_torgo('/home/tim/Documents/Datasets/torgo/TORGO')
        else:
            ds = import_torgo('/work/herzig/datasets/torgo/TORGO')
    elif name.lower() == 'hu':
        if local:
            ds = import_hu('/home/tim/Documents/Datasets/hu_final')
        else:
            ds = import_hu('/work/herzig/datasets/hu_final')
    elif name.lower() == ('uaspeech' or 'ua'):
        if local:
            ds = import_ua('/home/tim/Documents/Datasets/uaspeech')
        else:
            ds = import_ua('/work/herzig/datasets/uaspeech')

    return ds
