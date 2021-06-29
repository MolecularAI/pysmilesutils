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
#     display_name: tester
#     language: python
#     name: tester
# ---

# %% [markdown]
# This notebook contains examples for the `SMILESTokenizer` class to illustrate its usage.

# %% [markdown]
# # Imports

# %%
# %load_ext autoreload
# %autoreload 2

import sys

#sys.path.append("..")

import pickle
import matplotlib.pyplot as plt
from pysmilesutils.tokenize import *
from pysmilesutils.analyze import analyze_smiles_tokens

# %% [markdown]
# # Basic Usage

# %% [markdown]
# ## `SMILESTokenizer`

# %% [markdown]
# The easiest way to initialize the `SMILESTokenizer` is to call the constructor with a list of tokens, used when parsing smiles, and a list of SMILES which are used to create the vocabulary.

# %%
smiles = ['ClCCCN1CCCC1', 'CI.Oc1ncccc1Br']

tokenizer = SMILESTokenizer(tokens=["Br", "Cl"], smiles=smiles)
print("Encoded SMILES:", tokenizer(smiles))
print("Vocabulary dictionary:", tokenizer.vocabulary)

# %% [markdown]
# The vocabulary will change based on the SMILES supplied since the tokenizer treats all single characters as tokens, except for the ones supplied in the token list.
#
# It is possible to add more tokens to an already initialized tokenizer, but this will require an update to the vocabulary. Which can be made by supplying a list of SMILES.

# %%
tokenizer.add_tokens(["CCCC", "CCC", "CC"], smiles=smiles)
print("Encoded SMILES:", tokenizer(smiles))
print("Vocabulary dictionary:", tokenizer.vocabulary)

# %% [markdown]
# In this instance we added different number of carbons as separate tokens. Note that the tokenizer prioritizes the first tokens in the list when parsing. Thus, if we hade reversed the list, `["CC", "CCC", "CCCC"]`, the token `"CCCC"` would never have been considered a token, since it is treated as two pairs of `"CC"`.

# %% [markdown]
# Another way to accomplish this is to use regular expressions.

# %%
regex_tokens = ["C+","c+"]
regex_tokenizer = SMILESTokenizer(
    tokens=["Br", "Cl"], 
    regex_token_patterns=regex_tokens,
    smiles=smiles
)
print("Encoded SMILES:", regex_tokenizer(smiles))
print("Vocabulary dictionary:", regex_tokenizer.vocabulary)

# %% [markdown]
# Here we have included two regular expression tokens: 1 or more `"C"`, and 1 or more `"c"`. Note that these are present in the vocabulary as tokens, not regular expressions.

# %% [markdown]
# ## `SMILESAtomTokenizer` 

# %% [markdown]
# There also exists a pre built extension of the `SMILESTokenizer` called `SMILESAtomTokenizer`. This tokenizer treats all atoms as tokens.

# %%
smiles=['CI.Oc1ncccc1Br', 'COC(=O)Cc1c(C)nn(Cc2ccc(C=O)cc2)c1C.[Mg+]Cc1ccccc1']
atom_tokenizer = SMILESAtomTokenizer(smiles=smiles)
print("Encoded SMILES:", atom_tokenizer(smiles))
print("Vocabulary dictionary:", atom_tokenizer.vocabulary)

# %% [markdown]
# Note that both `Mg` and `Br` are part of the vocabulary.

# %% [markdown]
# # Useful Functions

# %% [markdown]
# ## Tokenizer Vocabulary

# %% [markdown]
# We can also manually update a vocabulary for a tokenizer, for example if we have another set of smiles. This is accomplished using the function `create_vocabulary_from_smiles`.

# %%
smiles = ['ClCCCN1CCCC1', 'CI.Oc1ncccc1Br']

tokenizer = SMILESTokenizer(smiles=smiles)

print(tokenizer.vocabulary)

new_smiles = ['N#Cc1ccnc(CO)c1', 'O=C(CCCl)c1ccccc1']

tokenizer.create_vocabulary_from_smiles(new_smiles)

print(tokenizer.vocabulary)

# %% [markdown]
# Note here that the two vocabularys are different.

# %% [markdown]
# ## Token Statistics

# %% [markdown]
# There also exists a function in `pysmilesutils.analyze` which provides som statistics for tokens in a set of SMILES

# %%
# Load reactant SMILES fro the Pande dataset 
with open("data/pande_dataset.pickle", "rb") as f:
    ds = pickle.load(f)
    reactants = list(ds.reactants)
    
smiles_tokenizer = SMILESAtomTokenizer(smiles=reactants)

data_stats = analyze_smiles_tokens(smiles_tokenizer, reactants) 
print(data_stats.keys())

# %%
num_tokens = data_stats["num_tokens"]

_ = plt.bar(*num_tokens)

# %%
token_freq = data_stats["token_frequency"]
for token, n in zip(*token_freq):
    print(f"{token:4}{n}")

# %%
