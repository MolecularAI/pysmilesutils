import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pysmilesutils.datautils import (
    MultiDataset,
    PickledMultiDataset,
    _BlockDataset,
    _accessitem,
    _accesslen,
    StringDataset,
    SMILESDataset,
)

_data_len = 10
_data = torch.arange(4 * _data_len).view(4, -1)


class TestVariousDatasets:
    def test_stringdataset(self):
        # Test that the string dataset ensure equal length of lists"""
        with pytest.raises(Exception):
            StringDataset(range(10), range(11))
        # Test aligned and proper length
        std = StringDataset(range(10), range(10))
        assert(std[0][0] == std[0][1])
        assert(len(std) == 10)
        # Test proper length of output
        std = StringDataset(range(10), range(10), range(10))
        assert(len(std[0]) == 3)
        # Test possibility for mixed "listlike" data
        assert(StringDataset(range(10), ["A"] * 10, torch.zeros(10)))
        
    def test_smilesdataset(self):
        # Test the sorted indices tensor can get created
        smids = SMILESDataset(["AA", "AAA", "A"])
        assert(all(smids.sorted_indices == torch.tensor([2, 0, 1])))
        
    def test_block_dataset(self):
        """Tests that the `_BlockDataset` correctly samples blocks and that its
        length is correct.
        """
        data = torch.arange(100).view(2, -1)
        dataset = TensorDataset(data[0], data[1])
        block_size = 20
        block_dataset = _BlockDataset(dataset, block_size=block_size)

        assert len(block_dataset) == 3

        dataloader = DataLoader(dataset, batch_size=block_size)
        for idx, batch in enumerate(dataloader):
            block = block_dataset[idx]
            assert all([torch.equal(t1, t2) for t1, t2 in zip(block, batch)])


class TestAccess:
    def test_access_items(self):
        """Tests that different data structures are accessed correctly. This
        tests some nested data structures as well as the length of the data.
        """
        data_ints = [1, 2, 3, 4]
        data_floats = [float(i) for i in data_ints]
        data_strings = ["This", "is", "a", "list", "of", "strings"]
        data_tensor = torch.arange(10)

        data_tensors = [t for t in torch.arange(20).view(4, -1)]
        data_list_of_lists = [[1, 2, 3], ["a", "b", "c"]]

        data_tuple = ([1, 2, 3], [4, 5, 6])

        self.access_simple(data_ints)
        self.access_simple(data_floats)
        self.access_simple(data_strings)
        self.access_simple(data_tensor)

        self.access_nested(data_tensors)
        self.access_nested(data_list_of_lists)
        self.access_nested(data_tuple)

    def access_simple(self, data):

        samples = _accessitem(data, slice(0, None))
        data_size = _accesslen(data)

        assert data_size == len(data)

        if isinstance(data, torch.Tensor):
            assert torch.equal(data, samples)
        else:
            assert samples == data
            pass

    def access_nested(self, data_list):
        sample_list = _accessitem(data_list, slice(0, None))
        data_size = _accesslen(data_list)

        assert data_size == len(data_list[0])

        for samples, data in zip(sample_list, data_list):
            if isinstance(data, torch.Tensor):
                assert torch.equal(data, samples)
            else:
                assert samples == data


def create_datasets():
    """Partly initalizes three `MultiDataset`:s using lambda expressions. These
    can then be used by just supplying `repeats` and `shuffle` parameter.
    """
    multidataset = (MultiDataset, {"data_collection": _data})

    pickle_dir = f"{os.path.dirname(__file__)}/test_pickle"
    pickle_paths = [f"{pickle_dir}/{p}" for p in os.listdir(pickle_dir)]

    pickledataset = (
        PickledMultiDataset,
        {"data_path": pickle_dir, "directory": True, "sort_files": True},
    )
    pickledataset_dir = (
        PickledMultiDataset,
        {"data_path": pickle_paths, "directory": False, "sort_files": True},
    )

    return [multidataset, pickledataset, pickledataset_dir]


datasets = create_datasets()


class TestMultiDataset:
    @pytest.mark.parametrize("Dataset, kwargs", datasets)
    def test_multidataset_end_error(self, Dataset, kwargs):
        """Makes sure that an error is raised when trying to access data after
        after reaching the end of the dataset, as well as the step function
        is woring properly when called directly.
        """
        dataset = Dataset(**kwargs, repeats=False, shuffle=False)

        for _ in range(dataset.num_steps):
            dataset.step()

        with pytest.raises(TypeError):
            dataset[0]

    @pytest.mark.parametrize("Dataset, kwargs", datasets)
    def test_multidataset_iterator_finite(self, Dataset, kwargs):
        """Tests that a finite, non shuffled, dataset returns the correct
        elements, and that it can be used as an iterator.
        """
        dataset = Dataset(**kwargs, repeats=False, shuffle=False)

        for idx, _ in enumerate(dataset):
            data = dataset[:]
            assert torch.equal(data, _data[idx])

    @pytest.mark.parametrize("Dataset, kwargs", datasets)
    def test_multidataset_iterator_infinite(self, Dataset, kwargs):
        """Tests that when `repeats` is true no error are raised and that data
        cycles properly.
        """
        dataset = Dataset(**kwargs, repeats=True, shuffle=False)

        for idx, _ in zip(range(_data.shape[0] * 3), dataset):
            data = dataset[:]

            assert torch.equal(data, _data[idx % _data.shape[0]])

    @pytest.mark.parametrize("Dataset, kwargs", datasets)
    def test_multidataset_in_dataloader(self, Dataset, kwargs):
        """Makes sure that the `MultiDataset` works in the `DataLoader` work
        flow.
        """
        dataset = Dataset(**kwargs, repeats=True, shuffle=False)

        num_epochs = _data.shape[0] * 3
        batch_size = _data_len // 2
        dataloader = DataLoader(dataset, batch_size=batch_size)

        for epoch, _ in zip(range(num_epochs), dataset):
            data_true = _data[epoch % _data.shape[0]]
            for bdx, batch in enumerate(dataloader):
                batch_true = data_true[(batch_size * bdx): (batch_size * (bdx + 1))]
                assert torch.equal(batch, batch_true)

    @pytest.mark.parametrize("Dataset, kwargs", datasets)
    def test_multidataset_restart(self, Dataset, kwargs):
        """Makes sure that the `MultiDataset` can be reiterated several times.
        """
        dataset = Dataset(**kwargs, repeats=False, shuffle=False)

        for idx, _ in enumerate(dataset):
            data = dataset[:]

            assert torch.equal(data, _data[idx])

        for idx, _ in enumerate(dataset):
            data = dataset[:]

            assert torch.equal(data, _data[idx])

    @pytest.mark.parametrize("Dataset, kwargs", datasets)
    def test_multidataset_shuffle(self, Dataset, kwargs):
        """Tests that the multi datasets shuffle properly."""
        dataset = Dataset(**kwargs, repeats=False, shuffle=True)

        num_tests = 1000
        non_equal = False

        for _ in range(num_tests):
            data = []
            for idx, _ in enumerate(dataset):
                data.append(dataset[:])
                non_equal += torch.equal(dataset[:], _data[idx])

            data = sorted(data, key=lambda d: d[0])

            for idx in range(len(data)):
                assert torch.equal(data[idx], _data[idx])

        assert non_equal

    @pytest.mark.parametrize("Dataset, kwargs", datasets)
    def test_multidataset_shuffle_repeats(self, Dataset, kwargs):
        """Tests that the multi datasets shuffle properly with repeats."""
        dataset = Dataset(**kwargs, repeats=True, shuffle=True)

        num_tests = 1000
        non_equal = False

        for idx, _ in zip(range(num_tests), dataset):
            non_equal += torch.equal(dataset[:], _data[idx % len(_data)])

        assert non_equal
