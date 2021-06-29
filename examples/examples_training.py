# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pickle, sys, os

#sys.path.append("..")

from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem

from pysmilesutils.tokenize import SMILESTokenizer, SMILESAtomTokenizer
from pysmilesutils.analyze import analyze_smiles_tokens
from pysmilesutils.augment import SMILESAugmenter

from transformer import TransformerModel
from pysmilesutils.datautils import BucketBatchSampler

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

# Remove later
import pdb, random
from time import time
from statistics import stdev, mean
from tqdm.notebook import tqdm


# %%
# Chose CUDA device
get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')

# %% [markdown]
# # Data
# %% [markdown]
# Here we load data from disk into CPU memory once, for demonstration purposes.

# %%
data = pickle.load(open("data/pande_dataset.pickle","rb"))
print(data.columns)

# %% [markdown]
# Get list of inices sorted by length

# %%
lengths = torch.tensor(list(map(len, data.reactants)))
_, reactants_sorted_indices = torch.sort(lengths)


# %%
display(Chem.MolFromSmiles(data.reactants[0])) #reactants
display(Chem.MolFromSmiles(data.products[0])) #product

# %% [markdown]
# # Tokenizer
# %% [markdown]
# The `SMILESTokenizer` is a callable class that tokenizes SMILES. Here we use a subclass that treats all atoms as tokens. We start by using our data set to create the vocabulary, i.e., the set of tokens and indices of the all the SMILES. The idea is that later we should be a able to save a vocabulary for a dataset along with the data. 

# %%
tokenizer = SMILESAtomTokenizer()
tokenizer.create_vocabulary_from_smiles(data.reactants + data.products)

# %% [markdown]
# We will also supply the user with different analyzing tools, so that we can view different statistics of our data. Below we present a bar plot of the distribution of number of tokens of the SMILES.

# %%
dataset_info = analyze_smiles_tokens(tokenizer, list(data.reactants) + list(data.products))
plt.bar(dataset_info["num_tokens"][0], dataset_info["num_tokens"][1])
plt.axvline(max(dataset_info["num_tokens"][0]), color='r', label="max num tokens")
plt.legend()

# %% [markdown]
# The `tokenizer` returns a list of tensors that can then be passed on to functions such as `torch.nn.utils.rnn.pad_sequence`. By default this function returns a tensor with the batch index as the second index. This conforms to the input shape for `torch.nn.Transformer` but it can be overriden with `batch_first=False`.

# %%
smiles_encoded = tokenizer(data.reactants[:20])
plt.matshow(pad_sequence(smiles_encoded, batch_first=True))

# %% [markdown]
# The vocabulary is simply a python dictionary.

# %%
for t in sorted(tokenizer.vocabulary):
    print(t, end=", ")

# %% [markdown]
# # Augmenter
# %% [markdown]
# The user will be supplied with a base augmenter class called `Augmenter` that can be extended to implement different augmentations. The simples example is the randomization of SMILES, implemented in the subclass `SMILESAugmenter`. This class is also callable.

# %%
augmenter = SMILESAugmenter()

smi = data.reactants[1]
smi_aug = augmenter(smi)

print(f"Original SMILES:   {smi}")
print(f"Augmented SMILES: {smi_aug}")

# %% [markdown]
# # Model
# %% [markdown]
# Here we use an adopted version of the `torch.nn.transformer`, which includes encodings and feed forward networks. For mor details see `transformer.py` and [this guide](https://pytorch.org/tutorials/beginner/transformer_tutorial.html).

# %%
transformer = TransformerModel(
    n_tokens=len(tokenizer), 
    d_model=256, 
    nhead=8, 
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024, 
    dropout=0.1
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu" #Uncomment for debugging
batch_size = 64

transformer = transformer.to(device)

# %% [markdown]
# # Training Schemes
# %% [markdown]
# Here we present a number of different proposals for how the tokenizer and augmenter can be used. The idea is to make them flexible enough that they can fit different use cases.
# %% [markdown]
# #### Tokenizing in Training Loop
# %% [markdown]
# Here we simply apply the tokenizer and augmenter to the batched data inside the training loop. To do this we also require a simple Dataset object that returns our reactants and targets.

# %%
class SMILESDataset(Dataset):
    def __init__(self, reactants, products):
        self.reactants = reactants
        self.products = products

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, idx):
        return self.reactants[idx], self.products[idx]



# %%
dataset = SMILESDataset(list(data.reactants), list(data.products),)

tokenizer.batch_first = False
tokenizer.adaptive_padding = True
augmenter = SMILESAugmenter(restricted=True)

dataloader = DataLoader(
    dataset=dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, 
    drop_last=True,
    pin_memory=True,
) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters())

t = time()

device = next(transformer.parameters()).device

lengths = []

for src, tgt in tqdm(dataloader, total=len(dataloader)):  
    
    src = tokenizer(augmenter(src))
    tgt = tokenizer(augmenter(tgt))
    
    src = pad_sequence(src).to(device)
    tgt = pad_sequence(tgt).to(device)
    
    lengths.append(src.shape[0])

    optimizer.zero_grad()
    output = transformer(src, tgt)

    loss = criterion(output.transpose(0,1).transpose(1,2), tgt.transpose(0,1))
    loss.backward()

    optimizer.step()

t = time() - t
print(f"Epoch time: {t}")
print(f"Samples/sec: {len(dataloader) * dataloader.batch_size / t}")
print(f"Average batch: {mean(lengths)}")

plt.bar(list(range(len(lengths))), lengths)

# %% [markdown]
# This seems fine but we can do better.

# %% [markdown]
# ## Tokenizing in the Dataset
# %% [markdown]
# We can also make the tokenizer and augmenter part of the dataset. Keep in mind this will result in these function being called on a single sample at a time.

# %%
class TokenizingDataset(Dataset):
    def __init__(self, reactants, products, tokenizer, augmenter):
        self.reactants = reactants
        self.products = products
        
        self.tokenizer = tokenizer
        self.augmenter = augmenter

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, idx):
        src = self.reactants[idx]
        tgt = self.products[idx]
        
        src = self.tokenizer(self.augmenter(src))
        tgt = self.tokenizer(self.augmenter(tgt))
        
        return (src[0], tgt[0])
    
def collate_variable_length(data):
    src, tgt = zip(*data)
    return (pad_sequence(src), pad_sequence(src))

# %% [markdown]
# Our training loop will now look like below.

# %%
augmenter = SMILESAugmenter(restricted=True)

dataset = TokenizingDataset(
    list(data.reactants), 
    list(data.products),
    tokenizer=tokenizer, 
    augmenter=augmenter
)

dataloader = DataLoader(
    dataset=dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0, 
    drop_last=True,
    collate_fn=collate_variable_length,
    pin_memory=True,
) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters())

t = time()

device = next(transformer.parameters()).device

lengths = []
    
for src, tgt in tqdm(dataloader, total=len(dataloader)): 
    src = src.to(device)
    tgt = tgt.to(device)
        
    optimizer.zero_grad()
    output = transformer(src, tgt)
    
    lengths.append(src.shape[0])

    loss = criterion(output.transpose(0,1).transpose(1,2), tgt.transpose(0,1))
    loss.backward()

    optimizer.step()

t = time() - t
print(f"Epoch time: {t}")
print(f"Samples/sec: {len(dataloader) * dataloader.batch_size / t}")
print(f"Average batch: {mean(lengths)}")

# %% [markdown]
# ## Tokenizing in the Collate Funtion
# %% [markdown]
# We can also make the application of the tokenizer and augmenter part of a collate function that we enter as a paramter to the Dataloader. We define a collate class that is callable and contains the tokenizer and augmenter.
# %% [markdown]
# Reusing our previous dataset our training loop now looks like below.

# %%
class SMILESCollater():
    def __init__(self, tokenizer, augmenter):
        self.tokenizer = tokenizer
        self.augmenter = augmenter

    def __call__(self, data):
        # we also apply the default_collate function
        return self.__collate__(default_collate(data))

    def __collate__(self, data):
        output = [pad_sequence(self.tokenizer(self.augmenter(d))) for d in data]
        return tuple(output)



# %%
dataset = SMILESDataset(list(data.reactants), list(data.products),)

tokenizer.batch_first = False
tokenizer.adaptive_padding = True
augmenter = SMILESAugmenter(restricted=True)

collater = SMILESCollater(tokenizer, augmenter) # defining collater

dataloader = DataLoader(
    dataset=dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, 
    drop_last=True,
    pin_memory=True,
    collate_fn=collater,
) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters())

t = time()

device = next(transformer.parameters()).device

lengths = []

for src, tgt in tqdm(dataloader, total=len(dataloader)):  
    
    lengths.append(src.shape[0])
    
    src = src.to(device)
    tgt = tgt.to(device)
    
    optimizer.zero_grad()
    output = transformer(src, tgt)

    loss = criterion(output.transpose(0,1).transpose(1,2), tgt.transpose(0,1))
    loss.backward()

    optimizer.step()

t = time() - t
print(f"Epoch time: {t}")
print(f"Samples/sec: {len(dataloader) * dataloader.batch_size / t}")
print(f"Average batch: {mean(lengths)}")
plt.bar(list(range(len(lengths))), lengths)


# %% [markdown]
# This gave a speedup of the training. However we can try and make it even better. If the mini-batches are made out of SMILES of similar length, the padding will be minimized, and training will be faster. 

# %% [markdown]
# ## Minimum batch length and max perfomance 
# %% [markdown]
# Here we show how the performance is affected the number of tokens in each sample. To illustrate this we sort the list of SMILES by length, and then we turn off batch shuffling. By doing this we force samples in batches to be of similar length.

# %%
src_sorted, tgt_sorted = sorted(list(data.reactants), key=len), sorted(list(data.products), key=len)

sorted_dataset = SMILESDataset(src_sorted, tgt_sorted,)

plt.plot(list(map(len,src_sorted)))


# %%
token_list = sorted_dataset[:]


# %%
tokenizer.batch_first = False
tokenizer.adaptive_padding = True
augmenter = SMILESAugmenter(restricted=True)

collater = SMILESCollater(tokenizer, augmenter) # defining collater

dataloader = DataLoader(
    dataset=sorted_dataset, 
    batch_size=batch_size, 
    shuffle=False, # to force batches of similar size 
    num_workers=4, 
    drop_last=True,
    pin_memory=True,
    collate_fn=collater, # added collater as parameter
) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters())

t = time()

device = next(transformer.parameters()).device

lengths = []
for src, tgt in tqdm(dataloader, total=len(dataloader)):  
    lengths.append(src.shape[0])
    
    src = src.to(device)
    tgt = tgt.to(device)

    optimizer.zero_grad()
    output = transformer(src, tgt)

    loss = criterion(output.transpose(0,1).transpose(1,2), tgt.transpose(0,1))
    loss.backward()

    optimizer.step()

t = time() - t
print(f"Epoch time: {t}")
print(f"Samples/sec: {len(dataloader) * dataloader.batch_size / t}")
print(f"Average batch: {mean(lengths)}")

# %% [markdown]
# What a speedup! However, it's probably detrimental to training to use the same mini-batches over and over again.

# %% [markdown]
# ## Bucket batch sampler
# The bucket sampler tries to balance the need for minimized padding with the need for random mini-batches. It divides the samples into "buckets" of similar lengths, and then draws random mini-batches from each bucket. The mini-batches will not be entirely random, but in our experiments it doesn't seem to detrimental to training. 

# %%
dataset = SMILESDataset(list(data.reactants), list(data.products),)

augmenter = SMILESAugmenter(restricted=True)

collater = SMILESCollater(tokenizer, augmenter) # defining collater

bucket_sampler = BucketBatchSampler(
    dataset, 
    indices=reactants_sorted_indices, 
    batch_size=batch_size, 
    bucket_size=1024,
)

dataloader = DataLoader(
    dataset=dataset, 
    batch_sampler=bucket_sampler,
    num_workers=4, 
    pin_memory=True,
    collate_fn=collater, # added collater as parameter
) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters())

t = time()

device = next(transformer.parameters()).device

lengths = []

for src, tgt in tqdm(dataloader, total=len(dataloader)):  
    lengths.append(src.shape[0])
    
    src = src.to(device)
    tgt = tgt.to(device)
        
    optimizer.zero_grad()
    output = transformer(src, tgt)

    loss = criterion(output.transpose(0,1).transpose(1,2), tgt.transpose(0,1))
    loss.backward()

    optimizer.step()

t = time() - t
print(f"Epoch time: {t}")
print(f"Samples/sec: {len(dataloader) * dataloader.batch_sampler.batch_size / t}")
print(f"Average batch: {mean(lengths)}")
plt.bar(list(range(len(lengths))), lengths)

# %% [markdown]
# A bit longer training time, but actually quite close to the maximum observed with the sorted dataset.
# %%

# %%
