# PySMILESutils

PySMILES utilities is a package of tools for handling encoding and decoding of SMILES for deep learning applications in PyTorch. The package contains a flexible tokenizer that can be used to analyze a given SMILES dataset using regular expressions and build a vocabulary of tokens, which can subsequently be used to encode the molecules via SMILES into pytorch tensors.
The augment class can be used for data augmentation via SMILES enumeration or atom order randomization.

Moreover, the package contains a variety of dataset, sampler and dataloader classes for pytorch. These solve various tasks that can appear. The BucketBatchSampler devides the dataset into buckets, and randomly creates mini-batches from within each bucket. This way the mini-batches can be created of SMILES of approximate similar length and sequence padded can be kept at a minimum. This speeds up training.

For datasets that are too large to fit in memory, chunck based loading can be applied, and for data that needs pre-augmentation (e.g. slow Levenshtein augmentation), the epochs can be pre-created on disk.


## Prerequisites

Before you begin, ensure you have met the following requirements:

* Linux, Windows or macOS platforms are supported - as long as the dependencies are supported on these platforms.

* You have installed [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) with python 3.6 - 3.8

The tool has been developed on a Linux platform.


## Installation

### Dependencies

Depencies are listed in environment.yml file and can be installed in the conda environment, either during creation

```bash
conda env create -f environment.yml
```

or updating an already activated environment

```bash
conda env update --file environment.yml
```


### Installation with pip
```sh
git clone https://github.com/MolecularAI/pysmilesutils.git

cd pysmilesutils

pip install .
```

pip can also install directly from github

```
python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git
```

Alternativly, the package can also be installed in developer mode, which leaves the source directory editable and the package still instantly usable without the need to reinstall after every change.

```bash
pip install -e .
```

### Testing
Post-installation the package can be tested with pytest.

```bash
cd tests

pytest
```

It is also recommended to run through the scripts in the example directory.

## Documentation

Sphinx documentation can be build with e.g. the make.sh in the "docs" directory

```bash
./docs/make.sh
```

Moreover, the examples directory contains some #%% delimited notebooks that show how to use the various classes. These notebooks can be paired with jupyter notebooks using the jupytext extension, and is also VScode compatible. #%% delimited scripts are much more GIT friendly than jupyter notebooks.

The training example contains a full example on how to train a transformer model using different approaches for handling the conversion of the SMILES in the mini-batches.

## Contributing

We welcome contributions, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.


To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the remote branch: `git push`
5. Create the pull request.

Please use ``black`` package for formatting, and follow ``pep8`` style guide.

## Contributors

* Esben Jannik Bjerrum, esben.bjerrum@astrazeneca.com
* Samuel Genheden, samuel.genheden@astrazeneca.com
* Christos Kannas, christos.kannas@astrazeneca.com
* Tobias Rastemo, tobias.rastemo@gmail.com

## License

The software is licensed under the Apache 2.0 license (see LICENSE file), and is free and provided as-is.

## References

Framework:

* Bjerrum, E., Rastemo, T., Irwin, R., Kannas, C. & Genheden, S. PySMILESUtils – Enabling deep learning with the SMILES chemical language. ChemRxiv (2021). [doi:10.33774/chemrxiv-2021-kzhbs](https://doi.org/10.33774/chemrxiv-2021-kzhbs)

Augmentation:

* Bjerrum, E. SMILES Enumeration as Data Augmentation for Neural Network Modeling of Molecules. Arxiv (2017) [https://arxiv.org/abs/1703.07076](https://arxiv.org/abs/1703.07076)



