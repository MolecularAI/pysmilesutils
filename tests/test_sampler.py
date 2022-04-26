import pytest
import torch

from pysmilesutils.datautils import (
    BucketBatchSampler,
    _SubsetSequentialSampler,
    TokenSampler,
)

n = 1000
data = torch.arange(n)


class TestSamplers:
    def test_bucket_sampler_errors(self):
        """Make sure that the `BucketBatchSampler` throws errors when neither
        bucket size nor number of bockets is given as well as when both are given.
        """
        with pytest.raises(ValueError):
            _ = BucketBatchSampler(
                data,
                batch_size=1,
            )

        with pytest.raises(ValueError):
            _ = BucketBatchSampler(
                data,
                bucket_size=100,
                num_buckets=10,
                batch_size=1,
            )

    def test_num_buckets(self):
        """Small test case that tests the number of batches produced by the sampler."""
        bucket_sampler = BucketBatchSampler(
            data,
            num_buckets=11,
            batch_size=50,
            drop_last=False,
        )
        assert len(bucket_sampler) == 22

    def test_bucket_len_drop_true(self):
        """Makes sure that batches get dropped when `drop_last` is `True`."""
        bucket_sampler = BucketBatchSampler(
            data,
            bucket_size=400,
            batch_size=300,
            drop_last=False,
        )
        assert len(bucket_sampler) == 5

    def test_bucket_len_drop_false(self):
        """Makes sure that the sampler produces the correct number of batches
        when `drop_true` is `False`.
        """
        bucket_sampler = BucketBatchSampler(
            data,
            bucket_size=400,
            batch_size=300,
            drop_last=True,
        )
        assert len(bucket_sampler) == 2

    def test_bucket_samples_all(self):
        """Tests that all data is sampled when shuffling and not droping."""
        bucket_sampler = BucketBatchSampler(
            data,
            bucket_size=400,
            batch_size=13,
            drop_last=False,
        )
        samples = []
        for s in bucket_sampler:
            samples.extend(s)

        assert torch.equal(torch.tensor(sorted(samples)), data)

    def test_subset_sequential_sampler(self):
        """Makser sure that the sequential sampler just gives the indices in a
        list or range.
        """
        r = range(1000)
        sampler = _SubsetSequentialSampler(r)
        samples = [s for s in sampler]

        assert samples == list(r)

        data = [1, 4, 6, 2, 45, 2, 8, 0, 8, 4, 2, 3, 4, 12]
        sampler = _SubsetSequentialSampler(data)
        samples = [s for s in sampler]

        assert samples == data

    def test_token_sampler(self, mocker):
        def fake_choices(*args, **kwargs):
            weights = kwargs["weights"]
            for idx, weight in enumerate(weights):
                if weight > 0:
                    return [idx]
            return [0]

        mocked_choices = mocker.patch(
            "pysmilesutils.datautils.random.choices", side_effect=fake_choices
        )
        lengths = []
        for idx in range(10):
            lengths.extend([idx + 10] * 50)

        sampler = TokenSampler(
            num_buckets=5, seq_lengths=lengths, batch_size=100, shuffle=False
        )

        batches = list(sampler)

        assert len(sampler) == 75
        assert all(len(bucket) == 100 for bucket in sampler.buckets)
        assert len(batches) == 75
        # Check a few batches
        assert batches[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert batches[-1] == [495, 496, 497, 498, 499]
