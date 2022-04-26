# -*- coding: utf-8 -*-
"""Classes for data augmentation of SMILES.
"""
from abc import abstractmethod
from random import shuffle
from typing import Any
from typing import Iterable
from typing import List
from typing import Union

import numpy as np
from rdkit import Chem

Mol = Chem.Mol


class Augmenter:
    """An abstract base class for molecular augmenters.

    The class has one method, `augment`, which is overriden by child classes.
    It is possible to call the class with either a list of molecules or a single
    molecules. This input will then be passed to `augment` and the augmented
    molecule(s) will be returned.
    The Boolean ".active" property can be set to toggle augmentation.

    :param active: Whether the augmentation should be active or not, defaults to True.
    :param augment_prob: if lower than 1, it is used to randomly turn-off augmentation on an individual basis
    """

    def __init__(self, active: bool = True, augment_prob: float = 1.0) -> None:
        self.active = active
        self.augment_prob = augment_prob

    def __call__(self, data: Union[Iterable[Any], Any]) -> List[Any]:
        """Augments either a list of Anys or a single molecule by making sure
        the input is put into a `List` and then passed to the `augment` function.

        :param data: Either a list of molecules or a single molecules to be augmented.

        :return: A list of augmented molecules.
        """
        # Str is Iterable but must be encapsulated (e.g. single SMILES string)
        if not isinstance(data, Iterable) or isinstance(data, str):
            data = [data]

        return self.augment(data)

    @abstractmethod
    def _augment(self, data: Iterable[Any]) -> List[Any]:
        raise NotImplementedError()

    def augment(self, data: Iterable[Any]) -> List[Any]:
        """
        Augment a given list

        :param data: a list of molecules to be augmented.
        :return: A list of augmented molecules.
        """
        if self.active:
            return self._augment(data)
        return list(data)


class MolAugmenter(Augmenter):
    """
    Augmenter that works on RDKit Mol objects
    """

    def randomize_mols_restricted(self, mols: Iterable[Mol]) -> List[Mol]:
        """Randomizes the atom ordering of a list of RDKit molecules (`rdkit.Chem.Mol`:s).

        :param mols: List of RDKit molecules to be augmented.
        :return:  List of augmented RDKit molecules.
        """
        return list(map(self.randomize_mol_restricted, mols))

    def randomize_mol_restricted(self, mol: Mol) -> Mol:
        """Randomize the atom ordering of a RDKit molecule (`rdkit.Chem.Mol`).

        :param mol:  RDKit molecule to get a randomized atom order.
        :return: RDKit molecule object with a randomized atom-order.
        """
        # Standard shuffle surprisingly leads to 35% slower code.
        if self.augment_prob < np.random.rand():
            return mol
        atom_order: List[int] = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_order)
        return Chem.RenumberAtoms(mol, atom_order)

    def _augment(self, data: Iterable[Mol]) -> List[Mol]:
        """Randomizes `RDKit molecules by shuffling the atom order.

        :param data: List of RDKit molecules to be randomized.
        :return:  A list of randomized molecules.
        """
        return self.randomize_mols_restricted(data)


class SMILESAugmenter(MolAugmenter):
    """An augmenter that produces Augmented SMILES. (aka. SMILES enumeration/SMILES Randomization)

    The ´SMILESAugmenter` can use either an unrestricted or a restricted scheme.
    In the former case the `rdkit` SMILES augmentation is used, and in
    the later the atom order in the RDKit molecule is randomized before producing the
    non-canonical SMILES. The unrestricted provides more SMILES per molecule, but also contains
    more complex branching and ring-closure patterns than the restricted version.

    :param active: Whether the augmentation should be active or not, defaults to True.
    :param augment_prob: if lower than 1, it is used to randomly turn-off augmentation on an individual basis
    :param restricted: Use restricted augmentation rather than fully randomized, defaults to True
    """

    def __init__(
        self, active: bool = True, augment_prob: float = 1.0, restricted: bool = True
    ) -> None:
        self.restricted = restricted
        super().__init__(active, augment_prob)

    def augment_smiles(self, smiles: Iterable[str]) -> List[str]:
        """Augments a list of SMILES using the RDKit SMILES doRandom flag.

        This scheme is referred to as unrestricted since it uses the RDKit doRandom
        method. For restricted randomization see `~augment_smiles_restricted`.


        :param smiles: List of SMILES to be augmented.
        :return:  List of augmented SMILES.
        """
        smiles_aug: List[str] = []
        for smi in smiles:
            if self.augment_prob < np.random.rand():
                smiles_aug.append(smi)
                continue

            mols: List[Mol] = list(map(Chem.MolFromSmiles, smi.split(".")))
            for _ in range(3):
                try:
                    smi_new: List[str] = [Chem.MolToSmiles(mol, doRandom=True) for mol in mols]
                    shuffle(smi_new)
                    smiles_aug.append(".".join(smi_new))
                    break
                except Exception as e:
                    print(f"Augmentation failed for {smi} with error: {e}")
            else:
                smiles_aug.append(smi)
                print(f"Augmentation failed three times for {smi}, returning unaugmented original")
                
        return smiles_aug

    def augment_smiles_restricted(self, smiles: Iterable[str]) -> List[str]:
        """Augments a list of SMILES using restricted atom ordering randomization.

        The restricted augmentation method randomizes the atom ordering of a
        RDKit molecule object before creating a non-canonical SMILES.
        If multiple molecules are present in the smiles (. separated),
        the order will be shuffled, but molecules with many atoms have a higher chance of being first.
        For an unrestricted SMILES augmentation see ~augment_smiles`.

        :param smiles: List of SMILES to be augmented.
        :return: List of augmented SMILES.
        """
        smiles_aug: List[str] = []

        augment_prob = self.augment_prob
        self.augment_prob = 1.0  # To avoid double application

        for smi in smiles:
            if augment_prob < np.random.rand():
                smiles_aug.append(smi)
                continue
            
            mol: Mol = Chem.MolFromSmiles(smi)
            for _ in range(3):
                try:
                    mol_rand = self.randomize_mol_restricted(mol)
                    smiles_aug.append(Chem.MolToSmiles(mol_rand, canonical=False))
                    break
                except Exception as e:
                    print(f"Augmentation failed for {smi} with error: {e}")
            else:
                smiles_aug.append(smi)
                print(f"Augmentation failed three times for {smi}, returning unaugmented original")

        self.augment_prob = augment_prob
        return smiles_aug

    def _augment(self, data: Iterable[str]) -> List[str]:
        """Augments a list of SMILES.

        The augmentation can be done either unrestricted using the
        SMILES doRandom of RDKit or restricted by randomizing the
        atom ordering before creating a non-canonical SMILES.

        :param data: List of SMILES to be augmented.
        :return: A list of augmented SMILES.
        """
        if self.restricted:
            return self.augment_smiles_restricted(data)

        return self.augment_smiles(data)
