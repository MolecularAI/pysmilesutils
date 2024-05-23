"""Data utility classes for PyTorch datasets.
"""
import math
import os
import pickle
import random
from itertools import count
from itertools import cycle
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Iterator
from typing import Sized

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
    BatchSampler,
    SubsetRandomSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.dataloader import default_collate

from pysmilesutils.augment import Augmenter
from pysmilesutils.tokenize import SMILESTokenizer


class StringDataset(Dataset):
    """Dataset wrapping list-like objects.

    Each sample will be retrieved by indexing lists.

    :param lists: One or more list-like objects that supports direct indexing
    """

    def __init__(self, *lists: List) -> None:
        assert all(len(lists[0]) == len(element) for element in lists)
        self.lists = lists

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        """Index the dataset

        :param index: index to be retrieced

        :return: Elements from the dataset that corresponds to the index
        """
        return tuple(element[index] for element in self.lists)

    def __len__(self) -> int:
        return len(self.lists[0])


class SMILESDataset(StringDataset):
    """SMILESDataset is a StringDataset with a sorted_indices property which is a list of
    indices sorted on the length of the first data list (needed for bucket sampling).
    """

    @property
    def sorted_indices(self) -> torch.Tensor:
        """
        Indices of the samples in the dataset sorted by the length of
        the content of the first entry in the dataset lists
        """
        lengths = torch.tensor(list(map(len, self.lists[0])))
        _, _sorted_indices = torch.sort(lengths)
        return _sorted_indices


class SMILESCollater:
    """Collater for SMILES strings with optional augmentation.
    If an augmenter is provided, it can be switched on and off by setting the augment property to True or False.

    :param tokenizer: The tokenizer to convert SMILES into Tensors
    :param augmenter: Augmenter to apply to the SMILES before tokenization, defaults to None
    """

    def __init__(self, tokenizer: SMILESTokenizer, augmenter: Augmenter = None) -> None:
        self.tokenizer = tokenizer
        self.augmenter = augmenter

        self.augment = bool(self.augmenter)

    def __call__(self, data: List[Any]) -> Tuple[torch.Tensor, ...]:
        # We also apply the default_collate function to handle batches when theres only single tensors in dataset samples
        return self._collate(default_collate(data))

    def _collate(self, data: List[Any]) -> Tuple[torch.Tensor, ...]:
        # No need to check for both self.augment & self.augmenter
        if self.augment and self.augmenter:
            output = [pad_sequence(self.tokenizer(self.augmenter(d))) for d in data]
        else:
            output = [pad_sequence(self.tokenizer(d)) for d in data]
        return tuple(output)


class _SubsetSequentialSampler(Sampler):
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class BucketBatchSampler(Sampler):
    """A sampler for bucketing samples before shuffling and batching.

    The data is split into `num_buckets` buckets. These buckets all contain
    consecutive data samples and are not in any way random. In order to sample
    from non-consecutive data the user can specify `indices` which is a list
    from which the samped indices are chosen instead. This means that a user
    can supply a list of indices corresponing to the data sorted by length, and
    buckets will then be of similar length.

    :param data_source: Data to generate sampled indices for.
    :param batch_size: The size of the batches in each bucket.
    :param indices: A sequence of indices to sample from
            instead of just the range of the data_source length. Defaults to None
    :param bucket_size: The number of buckets to
            partition data_source into. Mutually exclusive with `bucket_size`.
            Defaults to None.
    :param num_buckets: The number of buckets to
            partition data_source into. Mutually exclusive with `bucket_size`.
            Defaults to None.
    :param shuffle_batches:  Whether to shuffle batches
            internally in each bucket. Defaults to True.
    :param shuffle_buckets: Whether to shuffle from which
            bucket the next sample is drawn. Defaults to True.
    :param drop_last: Whether the last batch in each bucket
            can be dropped if it's below batch_size. Defaults to False.

    :raises ValueError: Raises error if both `bucket_size` and `num_buckets` is
            None, since one of these has to be specified.
    :raises ValueError: Raises error if both `bucket_size` and `num_buckets` are
            specified. These are mutually exclusive.
    """

    def __init__(
        self,
        data_source: Any,
        batch_size: int,
        indices: np.ndarray = None,
        bucket_size: int = None,
        num_buckets: int = None,
        shuffle_batches: bool = True,
        shuffle_buckets: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.data_source = data_source
        num_samples: int = len(data_source)

        if bucket_size and num_buckets:
            raise ValueError(
                "Only one of 'bucket size' and 'num_buckets' can be specified."
            )
        if bucket_size is None and num_buckets is None:
            raise ValueError(
                "One of 'bucket size' and 'num_buckets' must be specified."
            )
        if bucket_size is None:
            bucket_size = (num_samples + num_buckets - 1) // num_buckets
        elif num_buckets is None:
            num_buckets = (num_samples + bucket_size - 1) // bucket_size

        self.num_samples = num_samples
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets

        self.indices = indices

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_batches = shuffle_batches
        self.shuffle_buckets = shuffle_buckets

    def __iter__(self) -> Iterator[Any]:
        bucket_samplers = []

        sizes = [self.bucket_size] * self.num_buckets
        sdx = torch.randint(high=self.num_buckets, size=(1,))
        sizes[sdx] = self.bucket_size * (1 - self.num_buckets) + self.num_samples

        start_idx = 0

        for size in sizes:
            end_idx = start_idx + size

            if self.shuffle_batches:
                sampler: Sampler = SubsetRandomSampler(list(range(start_idx, end_idx)))
            else:
                sampler: Sampler = _SubsetSequentialSampler(
                    list(range(start_idx, end_idx))
                )

            batch_sampler: Sampler = BatchSampler(
                sampler=sampler,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            bucket_samplers.append(iter(batch_sampler))

            start_idx = end_idx

        while len(bucket_samplers) > 0:
            if self.shuffle_buckets:
                r = torch.randint(high=len(bucket_samplers), size=(1,))
            else:
                r = 0
            try:
                if self.indices is None:
                    yield next(bucket_samplers[r])
                else:
                    yield self.indices[next(bucket_samplers[r])].tolist()

            except StopIteration:
                del bucket_samplers[r]

    def __len__(self) -> int:
        num_batches_in_bucket = self.bucket_size / self.batch_size
        num_batches_in_small_bucket = (
            self.bucket_size * (1 - self.num_buckets) + self.num_samples
        ) / self.batch_size

        if self.drop_last:
            num_batches = (self.num_buckets - 1) * math.floor(
                num_batches_in_bucket
            ) + math.floor(num_batches_in_small_bucket)
        else:
            num_batches = (self.num_buckets - 1) * math.ceil(
                num_batches_in_bucket
            ) + math.ceil(num_batches_in_small_bucket)

        return num_batches


class _BlockDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        block_size: int = 10000,
    ) -> None:
        self.block_size = block_size
        self.dataset = dataset

    def __getitem__(self, idx: int) -> Any:
        start = idx * self.block_size
        end = min((idx + 1) * self.block_size, len(self.dataset))

        if start >= end:
            raise IndexError()

        return self.dataset[start:end]

    def __len__(self) -> int:
        return (len(self.dataset) + self.block_size - 1) // self.block_size


class _AccessDataset(Dataset):
    def __init__(self, data, dataset: Dataset) -> None:
        self.data = data
        self.dataset = dataset

    def __getitem__(self, idx: int) -> Any:
        try:
            return self.dataset._accessitem(self.data, idx)
        except AttributeError:
            return _accessitem(self.data, idx)

    def __len__(self) -> int:
        try:
            return self.dataset._accesslen(self.data)
        except AttributeError:
            return _accesslen(self.data)


def _accessitem(data, idx: int) -> Any:
    """This was inspired by the `default_collate` function.
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/
    """
    if isinstance(data, (tuple, list)):
        item = data[0]
        if not isinstance(item, (float, int, str)):
            return [d[idx] for d in data]

    return data[idx]


def _accesslen(data) -> int:
    """This was inspired by the `default_collate` function.
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/
    """
    if isinstance(data, (tuple, list)):
        item = data[0]
        if not isinstance(item, (float, int, str)):
            return len(item)

    return len(data)


class BlockDataLoader(DataLoader):
    """A DataLoader that loads data in blocks, and can apply shuffling within these blocks.

    The `BlockDataLoader` works by stacking two `DataLoaders` on top of each other.
    The outer `DataLoader` loads data using the supplied `Dataset` by retrieving data in blocks, instead of iteratively.
    This prevents non contiguous reads from disk. The inner `DataLoader` batches the data in each block.

    In order for the inner `DataLoader` to properly batch data the user needs to implement additional
    functions `_accessitem` and `_accesslen` in the `Dataset`.
    This is in addition to the implemented `__getitem__` function.
    The former of these functions needs to retrieve slices of data that has already been loaded into blocks.
    If no functions are specified a default access function is used.

    :param dataset: The `Dataset` used to retrieve data in the outer `DataLoader`.
    :param batch_size: The size of the batches retrieved by the inner `DataLoader`. Defaults to 100.
    :param block_size: The size of the blocks loaded by the outer `DataLoader`. Defaults to 1000.
    :param shuffle: Whether to shuffle internally in the blocks. Defaults to True.
    :param drop_last_batch: Whether to drop the last batch in each block if smaller than batch_size. Defaults to False.
    :param drop_last_block: Whether to drop the last block if it's smaller than the block_size. Defaults to False.
    :param collate_fn: Collate function that is applied to batches made from block data. Defaults to None.
    """

    def __init__(
        self,
        dataset: Dataset,
        *args,
        batch_size: int = 100,
        block_size: int = 1000,
        shuffle: bool = True,
        drop_last_batch: bool = False,
        drop_last_block: bool = False,
        collate_fn: Optional[Callable] = None,
        **kwargs
    ) -> None:
        self.inner_batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.drop_last_block = drop_last_block

        self.batch_collate_fn = collate_fn

        self.block_size = block_size
        self.shuffle = shuffle
        self.full_dataset = dataset

        block_dataset = _BlockDataset(
            dataset,
            block_size=block_size,
        )

        super().__init__(
            block_dataset,
            *args,
            shuffle=shuffle,
            collate_fn=lambda x: x,
            **kwargs,
        )

    def __iter__(self) -> Iterator[Any]:
        for block in super().__iter__():
            dataset = _AccessDataset(block[0], self.dataset)
            batch_loader = DataLoader(
                dataset=dataset,
                shuffle=self.shuffle,
                batch_size=self.inner_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last_batch,
                collate_fn=self.batch_collate_fn,
            )
            for batch in batch_loader:
                yield batch

    def __len__(self) -> int:
        num_samples = len(self.full_dataset)
        num_blocks = num_samples // self.block_size

        num_batches = self.block_size // self.inner_batch_size
        if not (self.drop_last_batch or self.block_size % self.inner_batch_size == 0):
            num_batches += 1

        num_tail_samples = (num_samples % self.block_size) * (not self.drop_last_block)
        num_tail_batches = num_tail_samples // self.inner_batch_size
        if not (self.drop_last_batch or num_tail_samples % self.inner_batch_size == 0):
            num_tail_batches += 1

        return num_blocks * num_batches + num_tail_batches


class MultiDataset(Dataset):
    """A `Dataset` that can cycled through to expose different data each step.

    The `MultiDataset` can be used to sample from different data each epoch.
    One function is exposed: `step`, which when called will step to the next
    set of data. It is also possible to use the `MultiDataset` as an iterator
    and step through it in a for loop.

    It is also possible to specify the parameters `repeats` and `shuffle`. The
    former makes it so that the `step` function iterates through the list of
    data infinitely, by repeating it. The latter specifies that the the order of
    data is randomized. If both are `True` sampling is done randomly with
    replacement infinitely.

    The default implementation is simply a list of different data, and the `step`
    function merely swtiches which item in the list is retrieved. For custom
    behaviour the function `step` can be overriden. Depending on the data the
    `__getitem__` function might have to be overriden as well, as the default
    implementation does not generalize to all data types.

    :param data_collection: List of different data to be cycled through.
    :param repeats: repeats (bool, optional): Whether to repeat the list of data once
            the end is reached. Defaults to False.
    :param shuffle: Whether to randomize the order data is
            exposed. Defaults to False.
    """

    def __init__(
        self,
        data_collection: List[Any],
        repeats: bool = True,
        shuffle: bool = False,
    ) -> None:
        self.repeats = repeats
        """Whether stepping of data is repeated once the last data is reached."""
        self.shuffle = shuffle
        """Whether the order of data is shuffled."""

        self._data_collection = data_collection

        self._reset()
        self.step()
        self.data = None

    def step(self) -> Any:
        """Steps the `MultiDataset` and changes from where `__getitem__` retrieves
        data.
        """
        try:
            self.__next__()
        except StopIteration:
            pass

    @property
    def num_steps(self) -> Union[int, float]:
        """The number of iterations of data in the `MultiDataset`."""
        if self.repeats:
            return float("inf")
        return len(self._data_collection)

    # TODO: is this method used?
    def _cycle(self, iterator, shuffle=False):
        while True:
            if shuffle:
                indices = torch.randperm(len(self._data_collection)).tolist()
            else:
                indices = range(len(self._data_collection))

            for idx in indices:
                yield self._getdata(idx)

    def _reset(self) -> None:
        if self.repeats and self.shuffle:
            indices = (
                random.choice(range(len(self._data_collection))) for _ in count()
            )
        elif self.repeats:
            indices = cycle(range(len(self._data_collection)))
        elif self.shuffle:
            indices = torch.randperm(len(self._data_collection))
        else:
            indices = range(len(self._data_collection))
        self.data_iterator = (self._getdata(idx) for idx in indices)

        self.data = None

    def _getdata(self, idx: int) -> Any:
        return self._data_collection[idx]

    def __getitem__(self, idx: int) -> Any:
        return _accessitem(self.data, idx)

    def __len__(self) -> int:
        return _accesslen(self.data)

    def __iter__(self) -> Iterator[Any]:
        self._reset()
        return self

    def __next__(self) -> Any:
        try:
            self.data = next(self.data_iterator)
        except StopIteration as err:
            self.data = None
            raise err


class PickledMultiDataset(MultiDataset):
    """An extension of the `MultiDataset` that cycles through data that is stored
    in different pickled files.

    Each time `step` is called the next set of data is loaded from disk into
    memory.

    :param data_path: Directory where the pickled files
            are stored or a list of file paths.
    :param directory:  Directory where the pickled files
            are stored or a list of file paths.
    :param repeats: Whether to repeat the list of data once
            the end is reached. Defaults to False.
    :param shuffle: [description], Whether to randomize the order data is
            exposed. Defaults to False.
    :param sort_files: Wheter to sort filenames, defaults to False

    :raises ValueError: If more than one element is given in data_path when using directory=True.
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        directory: bool = False,
        repeats: bool = False,
        shuffle: bool = False,
        sort_files: bool = False,
    ) -> None:
        if directory and isinstance(data_path, list):
            raise ValueError(
                "`data_path` can only contain one element when `directory` is `True`"
            )
        if directory:
            file_names = os.listdir(data_path)
            file_paths = [os.path.join(data_path, file) for file in file_names]
        elif isinstance(data_path, str):
            file_paths = [data_path]
        else:
            file_paths = data_path

        if sort_files:
            file_paths = sorted(file_paths)

        super().__init__(file_paths, repeats=repeats, shuffle=shuffle)

    def _getdata(self, idx: int) -> Any:
        with open(self._data_collection[idx], "rb") as f:
            return pickle.load(f)


class ChunkBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: Union[Sampler[int]],
        batch_size: int,
        drop_last: bool,
        i_chunk: int = 0,
        n_chunks: int = 1,
        **kwargs
    ) -> None:
        """
        A sampler which only samples a specific chunk of batches.

        :param sampler: The torch.Sampler used to sample data indices.
        :param batch_size: the number of samples in a batch.
        :param drop_last: whether to keep or drop the last batch (the last batch might
            be smaller than the other batches)
        :param i_chunk: the index of the current chunk of batches.
        :param n_chunks: the total number of chunks to divide the batches into.
        """
        super().__init__(sampler, batch_size, drop_last, **kwargs)
        self.i_chunk = i_chunk
        self.n_chunks = n_chunks
        self._batch_counter = 0
        self._set_start_end_batches()

    def __iter__(self) -> Iterator[List[int]]:
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in self.batch_size]
                    if self._batch_counter < self.start_batch_idx:
                        self._batch_counter += 1
                        continue

                    self._batch_counter += 1
                    if self._batch_counter >= self.end_batch_idx:
                        break
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * int(self.batch_size)
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    if self._batch_counter < self.start_batch_idx:
                        idx_in_batch = 0
                        batch = [0] * self.batch_size
                        self._batch_counter += 1
                        continue

                    self._batch_counter += 1
                    if self._batch_counter >= self.end_batch_idx:
                        break

                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
        self._batch_counter = 0

    def __len__(self):
        return self.end_batch_idx - self.start_batch_idx

    def _set_start_end_batches(self) -> None:
        """Divide batches into chunks of batches"""
        n_batches = int(np.ceil(len(self.sampler) / self.batch_size))
        n_batches_in_chunk = int(n_batches / float(self.n_chunks))
        self.start_batch_idx = self.i_chunk * n_batches_in_chunk
        if self.i_chunk != self.n_chunks - 1:
            self.end_batch_idx = self.start_batch_idx + n_batches_in_chunk
        else:
            self.end_batch_idx = n_batches


class TokenSampler(Sampler):
    """
    A Sampler which groups sequences into buckets based on length and constructs batches using
    a (potentially) different number of sequences from each bucket to achieve a target number of
    tokens in each batch. This approach has a number of advantages:
        - Faster training and eval since there are fewer pad tokens vs random batching
        - Potentially improved training stability since the number of tokens is approx the same
          each batch

    Note: There is a systematic error in the batch size (it will be slightly larger than the
          target size on average) since we simply take the mean of the seq lengths in the bucket,
          this does not account for padding that will result from the largest seq in the batch.

    :param num_buckets: Number of buckets to split sequences into
    :param seq_lengths: The length of the sequences in the dataset (in the same order)
    :param batch_size: Target number of tokens in each batch
    :param shuffle: Shuffle the indices within each bucket
    :param drop_last: Forget about the indices remaining at the end of each bucket
    """

    def __init__(
        self,
        num_buckets: int,
        seq_lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        if not drop_last:
            raise NotImplementedError("Keeping last elements is not yet supported")

        seq_lengths = np.asarray(seq_lengths)
        bucket_edges = np.histogram_bin_edges(
            seq_lengths,
            bins=num_buckets,
            range=(seq_lengths.min(), seq_lengths.max() + 1),
        )
        bucket_indices = np.digitize(seq_lengths, bins=bucket_edges)
        buckets = [
            np.where(bucket_indices == idx)[0].tolist()
            for idx in range(1, num_buckets + 1)
        ]
        lengths = [
            seq_lengths[np.where(bucket_indices == idx)[0]]
            for idx in range(1, num_buckets + 1)
        ]

        if shuffle:
            samplers = [RandomSampler(idxs) for idxs in buckets]
        else:
            samplers = [SequentialSampler(idxs) for idxs in buckets]

        # Work out approx number of sequences required for each bucket
        num_seqs = [batch_size // length.mean() for length in lengths]
        num_seqs = [int(num_sq) for num_sq in num_seqs]

        num_batches = [
            len(bucket) // num_seqs[b_idx] for b_idx, bucket in enumerate(buckets)
        ]
        num_batches = [int(num_bs) for num_bs in num_batches]

        self.num_seqs = num_seqs
        self.buckets = buckets
        self.num_batches = num_batches
        self.samplers = samplers

    def __iter__(self):
        iters = [iter(sampler) for sampler in self.samplers]
        rem_batches = self.num_batches[:]
        while sum(rem_batches) > 0:
            b_idx = random.choices(range(len(rem_batches)), weights=rem_batches, k=1)[0]
            batch_idxs = [next(iters[b_idx]) for _ in range(self.num_seqs[b_idx])]
            batch = [self.buckets[b_idx][idx] for idx in batch_idxs]
            rem_batches[b_idx] -= 1
            yield batch

    def __len__(self):
        return sum(self.num_batches)
