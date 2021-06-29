
# %%
# %load_ext autoreload
# %autoreload 2

import torch
import h5py
import time

from torch.utils.data import Dataset, TensorDataset, DataLoader
from pysmilesutils.datautils import BlockDataLoader, BucketBatchSampler, MultiDataset

from tqdm.notebook import tqdm


# %% [markdown]
# # `BlockDataLoader`

# %% [markdown]
# Det `BlockDataLoader` is used to split the data loading into two parts: first loading blocks, and then drawing batches from thse blocks. This can be usefull when datasets are very large and don't fit into memory.
#
# As an example lets look at data in the form of a single `torch.Tensor`. We use `BlockDataLoader` to load this in blocks of size `10`, from which we draw batches of size `5`.

# %%
class TestDataset(Dataset):
    data = torch.arange(20)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)


dataset = TestDataset()
    
dataloader = BlockDataLoader(dataset, batch_size=5, block_size=10)

for batch in dataloader:
    print(batch)


# %% [markdown]
# Note that all elements in a batch come from the same block, i.e., the numbers 0 through 9 are not mixed with the numbers 10 through 19.
#
# This is of course just a small example, and does not illstrate the real benefit of the `BlockDataLoader`. Lets instead look at an example with a larger dataset, stored on disk.

# %%
class HDFDataset(Dataset):
    """Small `Dataset` for loading HDF data."""
    def __init__(self, path):
        self.path = path
        
    def __len__(self):
        with h5py.File(self.path, "r") as f:
            num = len(f.get("tensor_data"))
        return num
    
    def __getitem__(self, idx):
        with h5py.File(self.path, "r") as f:
            data = f.get("tensor_data")[idx]
        return data
    

hdf_dataset = HDFDataset("data/data.hdf5")

# %% [markdown]
# Here we have created a dataset that loads tensor data stored in a HDF5 file. The file, `"data.hdf5"`, contains 1 million integer, 0 through 999999. Lets compare data loading using a block loader, and a torch dataloader. Below we load the entire dataset to memory for comparison.

# %%
with h5py.File("data/data.hdf5", "r") as f:
    # Loaded data with `h5py` is numpy arrays, we convert to torch tensors
    data_h5py = torch.tensor(f.get("tensor_data")[:])
    
data_h5py[:15]

# %% [markdown]
# Below we calculate the time it takes to load and shuffle the dataset using the `BlockDataLoader`. We also make sure that data is shuffled, and that all data is loaded. Note that we load data in blocksof 50000 samples. This means that we only shuffle batches within blocks of this size.

# %%
block_dataloader = BlockDataLoader(dataset=hdf_dataset, block_size=50000, batch_size=500)

data = []

for batch in tqdm(block_dataloader):
    data.extend(batch.tolist())

# Loaded data that has been shuffled
print(torch.equal(data_h5py, torch.tensor(data)))
# Loaded data that has been sorted
print(torch.equal(data_h5py, torch.tensor(sorted(data))))

# %%
dataloader = DataLoader(dataset=hdf_dataset, batch_size=500)

t = time.time()

for batch in tqdm(dataloader):
    pass

# %% [markdown]
# The time to load all batches using the `DataLoader` is significantly longer.

# %% [markdown]
# The `BlockDataLoader` receives several arguments which can alter its behaviour.

# %%
dataloader = BlockDataLoader(
    dataset=dataset,
    block_size=13,
    batch_size=4,
    drop_last_block=False,
    drop_last_batch=True,
    shuffle=False,
)

for batch in dataloader:
    print(batch)

# %% [markdown]
# Depending on how data is stored the default functions in `BlockDataLoader` might not be able to properly retrieve slices. In this case the user needs to specify the `_accessitem` and `_accesslen`.

# %% [markdown]
# # `BucketBatchSampler`

# %% [markdown]
# The `BucketBatchSampler` can be used to bucket items in the training set. This could, for example, be to make sure samples of similar length are passed to the model.

# %%
# random data of differentlenths
data = [
    torch.arange(torch.randint(1, 5, size=(1,)).item())
    for _ in range(20)
]


class WrapperDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    

def collate(data):
    return data


dataset = WrapperDataset(data)

_, sorted_indices = torch.sort(torch.tensor([len(d) for d in data]))
bucket_sampler = BucketBatchSampler(
    data,
    batch_size=3,
    num_buckets=3,
    indices=sorted_indices,
    drop_last=True,
)

dataloader = DataLoader(
    dataset,
    batch_sampler=bucket_sampler,
    collate_fn=lambda x: x,  # needed to not put batches into tensors
)

for batch in dataloader:
    print(batch)

# %% [markdown]
# Note that in each batch the tensors are of similar length

# %% [markdown]
# # `MultiDataset`

# %% [markdown]
# The `MultiDataset` can be used to iterate through different datasets each epoch. This can be usefull when a lot of data is present. As a dummy example let us look at a set of torch tensors.

# %%
# each list in the element represents one dataset
data_list = [
    torch.arange(start=(5 * idx), end=(5 * (idx + 1)))
    for idx in range(4)
]

dataset = MultiDataset(data_list, repeats=False, shuffle=False)

for _ in range(dataset.num_steps):
    print(dataset[:])
    dataset.step()

# %% [markdown]
# Here we used the `step` function to iterate through the different datasets. We could also use the multi dataset as an iterator

# %%
for _ in dataset:
    print(dataset[:])

# %% [markdown]
# We can also repeat data, to allow for an arbitrary number of epochs.

# %%
dataset = MultiDataset(data_list, repeats=True, shuffle=False)

num_epochs = 10

for _ in range(num_epochs):
    print(dataset[:])
    dataset.step()

# %% [markdown]
# We can also shuffle the dataset order, with our without repeats.

# %%
dataset = MultiDataset(data_list, repeats=False, shuffle=True)

for _ in dataset:
    print(dataset[:])

# %%
dataset = MultiDataset(data_list, repeats=True, shuffle=True)
    
for _ in range(num_epochs):
    print(dataset[:])
    dataset.step()

# %%

# %%
