import os

import pytest
from rdkit import Chem

from pysmilesutils.augment import SMILESAugmenter, MolAugmenter


class TestRandomizer:
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

    def get_num_new_random_smiles(self, test_smiles, smiles):
        num_new = 0

        for smi, smi_rand in zip(test_smiles, smiles):
            if smi != smi_rand:
                num_new += 1

        return num_new

    def test_smiles_majority_random_unrestricted(self, get_test_smiles):
        """Checks the `SMLIESRandomizer` by testing that when `restricted` is
        `False` mostly (99%) of the SMILES randomized are distinct from the
        canonical.
        """
        smiles_randomizer_unrestricted = SMILESAugmenter(restricted=False)
        randomized_smiles = smiles_randomizer_unrestricted(get_test_smiles)

        num_new = self.get_num_new_random_smiles(get_test_smiles, randomized_smiles)

        assert num_new / len(get_test_smiles) > 0.99

    def test_smiles_majority_random_restricted(self, get_test_smiles):
        """Checks the `SMLIESRandomizer` by testing that when `restricted` is
        `True` mostly (99%) of the SMILES randomized are distinct from the
        canonical.
        """
        smiles_randomizer_restricted = SMILESAugmenter(restricted=True)
        randomized_smiles = smiles_randomizer_restricted(get_test_smiles)

        num_new = self.get_num_new_random_smiles(get_test_smiles, randomized_smiles)

        assert num_new / len(get_test_smiles) > 0.99

    def test_mol_majority_random(self, get_test_smiles):
        """Checks the `MolRandomizer` by testing that mostly (99%) of the Mols
        randomized are distinct from the original canonical.
        """
        mol_randomizer = MolAugmenter()

        mols = [Chem.MolFromSmiles(smi) for smi in get_test_smiles]

        mols_randomized = mol_randomizer(mols)

        randomized_smiles = [
            Chem.MolToSmiles(mol, canonical=False) for mol in mols_randomized
        ]

        num_new = self.get_num_new_random_smiles(get_test_smiles, randomized_smiles)

        assert num_new / len(mols) > 0.99

    def test_mol_equality_random(self, get_test_smiles):
        """Check molecular equivalence after randomization by canonicalizing"""
        smiles_randomizer_unrestricted = SMILESAugmenter(restricted=False)
        randomized_smiles = smiles_randomizer_unrestricted(get_test_smiles)

        assert all(
            [
                Chem.MolToSmiles(Chem.MolFromSmiles(mol1))
                == Chem.MolToSmiles(Chem.MolFromSmiles(mol2))
                for mol1, mol2 in zip(randomized_smiles, get_test_smiles)
            ]
        )

    def test_mol_equality_restricted(self, get_test_smiles):
        """Check molecular equivalence after randomization by canonicalizing"""
        smiles_randomizer_unrestricted = SMILESAugmenter(restricted=True)
        randomized_smiles = smiles_randomizer_unrestricted(get_test_smiles)

        assert all(
            [
                Chem.MolToSmiles(Chem.MolFromSmiles(mol1))
                == Chem.MolToSmiles(Chem.MolFromSmiles(mol2))
                for mol1, mol2 in zip(randomized_smiles, get_test_smiles)
            ]
        )

    def test_active(self, get_test_smiles):
        """Tests that the `active` property works, i.e, that when the augmenter is
        not active it just returns the object that is input.
        """
        randomizer = SMILESAugmenter()
        smiles_rand = randomizer(get_test_smiles)

        assert smiles_rand != get_test_smiles

        randomizer.active = False
        smiles_nonrand = randomizer(get_test_smiles)

        assert smiles_nonrand == get_test_smiles

    def test_mol_low_aug_prob(self, get_test_smiles):
        """Check that by setting a very low augment probability few new SMILES are generated"""
        smiles_randomizer_unrestricted = SMILESAugmenter(
            restricted=False, augment_prob=0.1
        )
        randomized_smiles = smiles_randomizer_unrestricted(get_test_smiles)

        num_new = self.get_num_new_random_smiles(get_test_smiles, randomized_smiles)
        assert num_new / len(randomized_smiles) < 0.2
        assert num_new >= 1

    def test_mol_zero_aug_prob(self, get_test_smiles):
        """Check that by setting augment probability to zero, no new SMILES are generated"""
        smiles_randomizer_unrestricted = SMILESAugmenter(
            restricted=False, augment_prob=0.0
        )
        randomized_smiles = smiles_randomizer_unrestricted(get_test_smiles)

        num_new = self.get_num_new_random_smiles(get_test_smiles, randomized_smiles)
        assert num_new == 0

    def test_mol_zero_aug_prob_restricted(self, get_test_smiles):
        """Check that by setting augment probability to zero, no new SMILES are generated"""
        smiles_randomizer_unrestricted = SMILESAugmenter(
            restricted=True, augment_prob=0.0
        )
        randomized_smiles = smiles_randomizer_unrestricted(get_test_smiles)

        num_new = self.get_num_new_random_smiles(get_test_smiles, randomized_smiles)
        assert num_new == 0
