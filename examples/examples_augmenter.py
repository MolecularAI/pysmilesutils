# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: pysmilesutils
#     language: python
#     name: pysmilesutils
# ---

# %% [markdown]
# This notebook contains examples for the `Augmenter` class to illustrate its usage.

# %% [markdown]
# # Imports

# %%
# %load_ext autoreload
# %autoreload 2

import sys

#sys.path.append("..")

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from pysmilesutils.augment import *

# %% [markdown]
# # Basic Usage

# %% [markdown]
# The class `Augmenter` is a base class for augmenters. Child classes override the function `__augment__` which is used when augmenting objects.
#
# Two implementations are implemented by default: `SMILESRandomizer` and `MolRandomizer`. These can be used like below.

# %%
smiles_randomizer = SMILESAugmenter(restricted=True)
mol_randomizer = MolAugmenter()

# %% [markdown]
# The keyword `restricted` decides if a restricted SMILES randomization scheme is used, or the `rdkit.Chem.MolToSmiles`.

# %%
smiles = ['ClCCCN1CCCC1', 'CI.Oc1ncccc1Br']

print(smiles)
for _ in range(5):
    smiles_rand = smiles_randomizer(smiles)
    print(smiles_rand)

# %%
mols = [Chem.MolFromSmiles(smi) for smi in smiles]

print([Chem.MolToSmiles(mol, canonical=False) for mol in mols])
for _ in range(5):
    mols_rand = mol_randomizer(mols)
    print([Chem.MolToSmiles(mol, canonical=False) for mol in mols_rand])

# %%
