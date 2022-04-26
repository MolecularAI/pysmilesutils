import os

import pytest

from rdkit import Chem

from pysmilesutils.tokenize import SMILESTokenizer
from pysmilesutils.augment import SMILESAugmenter, MolAugmenter
from pysmilesutils.datautils import SMILESCollater
from pysmilesutils.datautils import SMILESDataset


class TestCollater:
    @pytest.fixture
    def get_test_smiles(self):
        try:
            current_dir = os.path.dirname(__file__)
            with open(os.path.join(current_dir, "./test_smiles.smi")) as file:
                smiles_data = file.readlines()
                test_smiles = [smi[:-1] for smi in smiles_data]
        except FileNotFoundError:
            print("Cannot find 'test_smiles.smi'")
        
        return test_smiles

    @pytest.fixture
    def smiles_tokenizer(self, get_test_smiles):
        return SMILESTokenizer(smiles=get_test_smiles)

    @pytest.fixture
    def unrestricted_augmenter(self):
        return SMILESAugmenter(restricted=False)

    @pytest.fixture
    def smids_single(self, get_test_smiles):
        return SMILESDataset(get_test_smiles)

    @pytest.fixture
    def smids_dual(self, get_test_smiles):
        return SMILESDataset(get_test_smiles, get_test_smiles)

    @pytest.fixture
    def get_random_smiles(self, get_test_smiles):
        mol_augmenter = MolAugmenter()
        mols = [Chem.MolFromSmiles(smi) for smi in get_test_smiles]
        randomized_mols = mol_augmenter(mols)
        randomized_smiles = [
            Chem.MolToSmiles(mol, canonical=False) for mol in randomized_mols
        ]

        return randomized_smiles

    @pytest.fixture
    def random_smids_single(self, get_random_smiles):
        return SMILESDataset(get_random_smiles)

    def test_single_input(self, smiles_tokenizer, smids_single):
        # Single input
        collater = SMILESCollater(tokenizer=smiles_tokenizer, augmenter=None)
        test_batch = collater([smids_single[i] for i in [1, 2, 3]])
        # Test that dims are as expected (first may depend on augmentation)
        assert(len(test_batch) == 1)
        assert(test_batch[0].shape[0] == 74)
        assert(test_batch[0].shape[1] == 3)

    def test_dual_input(self, smiles_tokenizer, smids_dual):
        # Dual inputs
        collater = SMILESCollater(tokenizer=smiles_tokenizer, augmenter=None)
        test_batch = collater([smids_dual[i] for i in [1, 2, 3]])
        # Test that second tensor is created and that shapes are as expected
        assert(len(test_batch) == 2)
        assert(test_batch[0].shape[0] == 74)
        assert(test_batch[0].shape[1] == 3)
        assert(test_batch[0].shape == test_batch[1].shape)

    def test_single_input_random(self, smiles_tokenizer, random_smids_single):
        # Single input
        collater = SMILESCollater(tokenizer=smiles_tokenizer, augmenter=None)
        test_batch = collater([random_smids_single[i] for i in [1, 2, 3]])
        # Test that dims are as expected (first may depend on augmentation)
        assert(len(test_batch) == 1)
        assert(test_batch[0].shape[0] > 0)
        assert(test_batch[0].shape[1] == 3)

    def test_augmentation_single(self, smiles_tokenizer, unrestricted_augmenter, smids_single):
        # Test augmentation for single input
        collater = SMILESCollater(tokenizer=smiles_tokenizer, augmenter=unrestricted_augmenter)
        test_batch = collater([smids_single[i] for i in [1, 2, 3]])

        assert(len(test_batch) == 1)
        assert(test_batch[0].shape[0] > 0)
        assert(test_batch[0].shape[1] == 3)

    def test_augmentation_double(self, smiles_tokenizer, unrestricted_augmenter, smids_dual):
        # Test augmentation for dual input
        collater = SMILESCollater(tokenizer=smiles_tokenizer, augmenter=unrestricted_augmenter)
        test_batch = collater([smids_dual[i] for i in [1, 2, 3]])

        assert(len(test_batch) == 2)
        assert(test_batch[0].shape[0] > 0)
        assert(test_batch[1].shape[0] > 0)
        assert(test_batch[0].shape[0] != test_batch[1].shape[0])
        assert(test_batch[0].shape[1] == test_batch[1].shape[1])

    def test_augmentation_single_random(self, smiles_tokenizer, unrestricted_augmenter, random_smids_single):
        # Test augmentation for single input
        collater = SMILESCollater(tokenizer=smiles_tokenizer, augmenter=unrestricted_augmenter)
        test_batch = collater([random_smids_single[i] for i in [1, 2, 3]])

        assert(len(test_batch) == 1)
        assert(test_batch[0].shape[0] > 0)
        assert(test_batch[0].shape[1] == 3)
