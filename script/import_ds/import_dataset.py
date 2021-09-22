from .import_hu import import_hu
from .import_torgo import import_torgo
from .import_mls_de import import_mls_de
from .import_mls_en import import_mls_en
from .import_cv import import_cv


# returns a list of datasets, if len == 1, then speaker independent, if len > 1 then speaker dependent
def import_dataset(name, local):
    ds = []
    
    if name.lower() == 'torgo':
        if local:
            ds = import_torgo('/home/tim/Documents/Datasets/torgo/test')
        else:
            ds = import_torgo('/work/herzig/datasets/torgo/TORGO')
    elif name.lower() == 'hu':
        if local:
            ds = import_hu('/home/tim/Documents/Datasets/hu_final')
        else:
            ds = import_hu('/work/herzig/datasets/hu_final')
    elif name.lower() == 'mls_de':
        if local:
            ds = import_mls_de(
                '/home/tim/Documents/Datasets/mls_german_opus/test/segments.txt',
                '/home/tim/Documents/Datasets/mls_german_opus/test/transcripts.txt',
                '/home/tim/Documents/Datasets/mls_german_opus/test/audio/')
        else:
            ds = import_mls_de(
                '/work/herzig/datasets/mls_german_opus/test/segments.txt',
                '/work/herzig/datasets/mls_german_opus/test/transcripts.txt',
                '/work/herzig/datasets/mls_german_opus/test/audio/')
    elif name.lower() == 'mls_en':
        ds = import_mls_en('10')
    elif name.lower() == 'cv_de':
        ds = import_cv('de', '10')
    elif name.lower() == 'cv_en':
        ds = import_cv('en', '10')

    return ds
