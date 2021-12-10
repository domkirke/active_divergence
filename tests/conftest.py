from active_divergence.data.audio import transforms
import pytest, itertools

DATASET_TESTS = ["acidsInstruments-test"]
DATASET_EXPORT_TRANSFORMS = [('none', None),
                             ('cqt', transforms.CQT()),
                             ('nsgt', transforms.NSGT())]

# configure inversion test
TRANSFORMS_INVERSION = []
winsizes = [512]
overlaps = [4]
normalize_args =[{'scale': 'unipolar', 'mode':'minmax'}, {'scale': 'unipolar', 'mode':'gaussian'},
                 {'scale': 'bipolar', 'mode': 'minmax'}, {'scale': 'bipolar', 'mode': 'gaussian'}]
contrasts = ['log', 'log1p']
mono_transform = transforms.Mono()
spectral_transforms = [('stft', transforms.STFT)]
# STFT
# for ws, ov, t in itertools.product(winsizes, overlaps, spectral_transforms):
#     stft_transform = t[1](ws, hop_size=int(ws / ov))
#     for n_args, contrast in itertools.product(normalize_args, contrasts):
#         name = f"{t[0]}-{ws}-{ov}-{n_args['mode'][0]}_{n_args['scale'][0]}_{contrast}"
#         mag_transform = transforms.Magnitude(normalize=n_args, contrast=contrast)
#         transform = transforms.ComposeAudioTransform([stft_transform, mag_transform])
#         TRANSFORMS_INVERSION.append((name+'_mag', mono_transform + transform))
#         polar_transform = transforms.Polar(mag_options={'normalize': n_args, 'contrast':contrast})
#         transform = transforms.ComposeAudioTransform([stft_transform, polar_transform])
#         TRANSFORMS_INVERSION.append((name+'_polar', mono_transform + transform))
#         polar_transform = transforms.PolarInst(mag_options={'normalize': n_args, 'contrast':contrast})
#         transform = transforms.ComposeAudioTransform([stft_transform, polar_transform])
#         TRANSFORMS_INVERSION.append((name+'_if', mono_transform + transform))

ds_rates = [50]
for dwn, n_args, contrast in itertools.product(ds_rates, normalize_args, contrasts):
    stft_transform = transforms.NSGT(ls=3.0, downsample=dwn)
    name = f"nsgt-{dwn}-{n_args['mode'][0]}_{n_args['scale'][0]}_{contrast}"
    mag_transform = transforms.Magnitude(normalize=n_args, contrast=contrast)
    transform = transforms.ComposeAudioTransform([stft_transform, mag_transform])
    TRANSFORMS_INVERSION.append((name + '_mag', mono_transform + transform))
    polar_transform = transforms.Polar(mag_options={'normalize': n_args, 'contrast': contrast})
    transform = transforms.ComposeAudioTransform([stft_transform, polar_transform])
    TRANSFORMS_INVERSION.append((name + '_polar', mono_transform + transform))
    polar_transform = transforms.PolarInst(mag_options={'normalize': n_args, 'contrast': contrast})
    transform = transforms.ComposeAudioTransform([stft_transform, polar_transform])
    TRANSFORMS_INVERSION.append((name + '_if', mono_transform + transform))


print(TRANSFORMS_INVERSION)

# TRANSFORMS_INVERSION = [TRANSFORMS_INVERSION[0]]

def pytest_generate_tests(metafunc):
    if 'dataset_path' in metafunc.fixturenames:
        metafunc.parametrize("dataset_path", DATASET_TESTS, scope="session")
    if 'dataset_export_transforms' in metafunc.fixturenames:
        metafunc.parametrize("dataset_export_transforms", DATASET_EXPORT_TRANSFORMS, scope="session")
    if 'transforms_inversion' in metafunc.fixturenames:
        metafunc.parametrize("transforms_inversion", TRANSFORMS_INVERSION, scope="session")

