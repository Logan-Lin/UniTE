# UniTE: A Survey and Unified Pipeline for Pre-training Spatio-temporal Trajectory Embeddings

Preprint Paper:

## Introduction
This is a unified and modular pipeline that aims to simplify the process of constructing and evaluating pre-training trajectory embedding methods.

## Pipeline Overview

![pipeline](./assets/pipeline.webp)

This pipeline modularizes pre-training methods into five key types of components: dataset, preprocessor, model, pre-training process, and downstream adaptor.

## Technical Overview

![tech-framework](./assets/tech-framework.png)

There are four main components of this framework:

- **Data storage**, including raw and pre-processed trajectory data, cached neural networks, training logs, and configuration files. There is also a dedicated class `data.Data` takes control of dataset loading and pre-processing.
- **Neural network models**, including samplers, encoders, and decoders. These models are the training and evaluation targets.
- **Pre-training**, which defines a specific pretext task and pre-trains the trajectory encoders and decoders.
- **Downstream task**, which applies the pre-trained encoders to trajectory mining tasks, so that the pre-training quality can be verified.

Following explains these components in detail.

### Data Storage

#### Raw dataset

The raw trajectory datasets stores the original trajectory sequences in `pandas HDF5Store` format. We provide a small, sample dataset in the `/sample` directory, that is also good for quick debugging.

One `HDF5` file stores one dataset, and there should be three keys (dataframes) stored in one file:

- `trips`, the trajectory sequences
- `trip_info`, additional features related to each trajectory, such as user ID and class label.
- `road_info`, information of the road network, such as coordinate of each road segment.

The `read_hdf` function in `Data` class loads the raw dataset and do some basic process.

#### Meta data

We store the original trajectories as `pandas DataFrame` as they are human-friendly, i.e., easy to read. Yet, they cannot be directly fed into neural networks and require extra pre-processing. The pre-processed and tensor-friendly arrays, such as trajectories padded to the same maximum length, are stored as `numpy` binaries. Once saved, we can quickly load them from files, saving times when we are conducting multiple sets of experiments.

The `dump_meta` and `load_meta` functions in the `Data` class save and load meta data from files, respectively.

#### Configurations

Configuration files control all the parameters in experiments. The config files are all stored in `/config` directory as the JSON format.

#### Cached models

We can save learnable parameters in neural networks as files. We can then load the models trained under a specific experimental setting, without the need to re-train them.

#### Base directory

All the above files, except for the configurations, are stored in a "base directory" specified in the `Data` class. If you are using sample datasets, the base directory will always be the `/sample` directory under the root of this framework. Otherwise, you should set a location with sufficient free space, since some of the above files will take up a lot of space!

### Models

The neural network models can be classified into:

- **Samplers**, re-sample the input trajectory sequences before passing them to the encoders. They come handy for achieving certain experimental settings such as denoising autoencoding. Although they are called "sampler", you can define how the trajectories are processed however you need.

- **Encoders**, encode and map trajectories into the embeddings. We already implement the common RNN- and Transformer-based encoders.
- **Decoders**, recover embeddings back to trajectories. They serve generative pre-training tasks such as auto-encoding.

It should be easy to implement new models, as long as you follow the input and output format. All model classes are located under the `/model` directory.

### Pre-training

#### Pre-trainer

The pre-trainers are classes that support the pre-training process. `/pretrain/trainer.py` includes commonly used pre-trainers. 

- The `Trainer` class is an abstract class, with implementation of the common functions: fetching mini-batches, feed mini-batches into loss functions, save and load the pre-trained models.
- The `ContrastiveTrainer` class is the trainer for contrastive-style pre-training. As for now, it doesn't include any additional function to the `Trainer` class.
- The `GenerativeTrainer` class is the trainer for generative-style pre-training. It includes a `generation` function that can be used to evaluate the generation accuracy of a trained encoder-decoder pair.
- The `MomentumTrainer` can be regarded as a special version of contrastive-style pre-trainer. It implements a momentum training scheme, with student-teacher pairs.
- The `NoneTrainer` is reserved for end-to-end training scenarios.

#### Pretext loss

The loss defines the pretext task to pre-train the models. Of coursely, they are aligned with the specified pre-trainer. For example, as a generative pretext loss, `AutoReg` can only work with `GenerativeTrainer`.

As their names suggest, `contrastive_losses.py` and `generative_losses.py` store contrastive- and generative-style loss functions. The loss functions have to obey two basic standards:

- They need to be a subclass of `torch.nn.Module`. This is because some loss functions may include extra discriminators or predictors.
- The `forward` function is the implementation of the loss's calculation.

We already include some widely used and SOTA pretext losses. For contrastive losses, we include the Maximum Entropy Coding loss and the InfoNCE loss. For generative losses, we include the Auto-regressive loss, and two Denoising Diffusion-based losses.

### Downstream tasks

We include four downstream tasks for evaluating the performance of pre-training representation methods. In `/downstream/trainer.py`, `Classification` class implements the classification task, `Destination` implements the destination prediction task, `Search` implements the similar trajectory search task, `TTE` implements the travel time estimation task.

You can also add your own tasks, just implement a new downstream trainer based on the abstract class `Trainer`. To add your own predictor for the downstream tasks, just add a new model class in `/downstream/predictor.py`.

## Contacts

If you have any questions, suggestions, or encounter any problems regarding the pipeline, feel free to contact me directly through my email:

[scholar@yanlincs.com](mailto:scholar@yanlincs.com)