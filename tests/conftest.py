from active_divergence.data.audio import transforms
import pytest

DATASET_TESTS = ["acidsInstruments-test"]
DATASET_EXPORT_TRANSFORMS = [('none', None),
                             ('cqt', transforms.CQT()),
                             ('nsgt', transforms.NSGT())]
TRANSFORMS_INVERSION = [('stft-512', transforms.STFT(512)),
                        ('stft-2048',transforms.STFT(2048)),
                        ('nsgt', transforms.NSGT())]



def pytest_generate_tests(metafunc):
    if 'dataset_path' in metafunc.fixturenames:
        metafunc.parametrize("dataset_path", DATASET_TESTS, scope="session")
    if 'dataset_export_transforms' in metafunc.fixturenames:
        metafunc.parametrize("dataset_export_transforms", DATASET_EXPORT_TRANSFORMS, scope="session")
    if 'transforms_inversion' in metafunc.fixturenames:
        metafunc.parametrize("transforms_inversion", TRANSFORMS_INVERSION, scope="session")

