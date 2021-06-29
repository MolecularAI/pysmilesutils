import torch
from torch.utils.data import TensorDataset, Dataset

from pysmilesutils.datautils import BlockDataLoader


class _ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class TestBlockDataLoader:
    def test_all_tensor_single(self):
        data = torch.arange(1000)
        dataset = TensorDataset(data)
        dataloader = BlockDataLoader(
            dataset,
            block_size=30,
            batch_size=100,
            shuffle=False,
            drop_last_batch=False,
            drop_last_block=False,
        )

        loaded_data = []
        for b in dataloader:
            loaded_data.extend(b[0].tolist())

        assert torch.equal(data, torch.tensor(loaded_data))

    def test_all_tensor_tuple(self):
        data = (torch.arange(1000), torch.arange(1000, 2000))
        dataset = TensorDataset(*data)
        dataloader = BlockDataLoader(
            dataset,
            block_size=30,
            batch_size=100,
            shuffle=False,
            drop_last_batch=False,
            drop_last_block=False,
        )

        loaded_data = [[], []]
        for t_1, t_2 in dataloader:
            loaded_data[0].extend(t_1.tolist())
            loaded_data[1].extend(t_2.tolist())

        for t, t_l in zip(data, loaded_data):
            assert torch.equal(t, torch.tensor(t_l))

    def test_string_list(self):
        data = list("abcdefgh")
        dataset = _ListDataset(data)

        dataloader = BlockDataLoader(
            dataset,
            block_size=3,
            batch_size=5,
            shuffle=False,
            drop_last_batch=False,
            drop_last_block=False,
        )

        loaded_data = []
        for s in dataloader:
            loaded_data.extend(s)

        assert data == loaded_data

    def test_shuffle(self):
        data = torch.arange(1000)
        dataset = TensorDataset(data)
        # torch.random.set_rng_state()
        dataloader = BlockDataLoader(
            dataset,
            block_size=30,
            batch_size=100,
            shuffle=True,
            drop_last_batch=False,
            drop_last_block=False,
        )

        loaded_data = []
        for b in dataloader:
            loaded_data.extend(b[0].tolist())

        assert not torch.equal(data, torch.tensor(loaded_data))

    def test_len(self):
        data = torch.arange(100)
        dataset = TensorDataset(data)
        dataloader = BlockDataLoader(
            dataset,
            block_size=40,
            batch_size=15,
            drop_last_batch=False,
            drop_last_block=False,
        )

        assert len(dataloader) == 8

        dataloader = BlockDataLoader(
            dataset,
            block_size=40,
            batch_size=15,
            drop_last_batch=False,
            drop_last_block=True,
        )

        assert len(dataloader) == 6

        dataloader = BlockDataLoader(
            dataset,
            block_size=40,
            batch_size=15,
            drop_last_batch=True,
            drop_last_block=False,
        )

        assert len(dataloader) == 5

        dataloader = BlockDataLoader(
            dataset,
            block_size=40,
            batch_size=15,
            drop_last_batch=True,
            drop_last_block=True,
        )

        assert len(dataloader) == 4
