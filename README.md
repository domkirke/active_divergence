# Active Divergence Toolbox

This toolbox is made for implementing, testing and scripting active divergence methods for audio, image and video generation algorithms. It is 
based on pytorch-lightning and hydra frameworks for pytorch programming, in order to allow for modulable and dynamic training.
The repository is organized as follows : 

- *active_divergence* : the repository containing the python package
- *configs*: the repository containg the configuration package to be combined with the hydra syntax
- *train_model.py* : the general code to train any active_divergence model 

## Install

```
git clone --recurse-submodules XXXXX
```

## General quickview
### Data
`active_divergence` can handle different data types, thanks to `hydra` configuration styles. Each data type is implemented in the `active_divergence/data` folder (for example, the available modes so far are `audio`, `image`, and `video`). Each data type has a `dataset.py` file, implementing the corresponding `Dataset` object, and a `module.py`, implementing the corresponding `LightningDataModule` object used by the trainer. This way, data import and models can be data-agnostic. For specific data types, see more details below.

### Models
Models are `pytorch_lightning.LightningModule` implementing high-level manipulations over big classes of architectures. Models also share general generation routines, in order to make manipulations / hacking general (then sometimes redundant). These methods are : 

- `full_forward` : transforms incoming input to generation
- `encode` : embeds incoming data to the latent embedding
- `decode` : generate data from a given input (latent, conditional...)
- `reconstruct` : reconstruct entierly an audio input
- `sample_from_prior` : sample generations from model's prior
- `trace` : trace model parameters and outputs from a given input (callback used for dissection)

<div style="text-align: center;">

| Model name       | Architectures             | Status |
| :--------------- | :------------------------ | :----- |
| `AutoEncoder`    | VAE, WAE                  | OK     |
| `GAN`            | GAN, WGAN, LSGAN, SpecGAN | OK     |
| `ProgressiveGAN` | PGAN, StyleGAN, GANSynth  | OK     |
| `ModulatedGAN`   | StyleGAN2, CatchAWaveform | OK     |
| `AutoRegressive` | SampleRNN, WaveNet        | OK     |

</div>





## Scripting

## Hacking 