import os
import pytest
import re
import torch

from pysmilesutils.tokenize import SMILESTokenizer, SMILESAtomTokenizer


class TestSMILESTokenizer:
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
    def tokenizer(self, get_test_smiles):
        return SMILESTokenizer(smiles=get_test_smiles)

    def test_default_arguments(self):
        with pytest.warns(Warning):
            tok = SMILESTokenizer()

        assert tok.vocabulary != {}
        assert tok.decoder_vocabulary != {}
        assert isinstance(tok.re, re.Pattern)

    def test_one_hot_encoding(self):
        smiles = ["BrC[nHCl]"]

        tok = SMILESTokenizer(smiles=sorted(list(smiles[0])), encoding_type="one hot")

        ids = tok.convert_tokens_to_ids(tok.tokenize(smiles))
        one_hot = tok(smiles)

        ids_truth = [1, 4, 11, 5, 7, 10, 6, 5, 9, 8, 2]
        one_hot_truth = torch.zeros(size=(11, 12))
        one_hot_truth[torch.arange(11), ids_truth] = 1

        assert torch.equal(one_hot[0], one_hot_truth)
        assert ids[0].tolist() == ids_truth

    def test_tokenize_no_tokens(self, get_test_smiles):
        tok = SMILESTokenizer(smiles=get_test_smiles)

        smiles_tokenized = tok.tokenize(get_test_smiles)

        assert smiles_tokenized == [
            ["^"] + list(smi) + ["&"] for smi in get_test_smiles
        ]

    def test_tokenize_multi_char_tokens(self):
        smiles = [
            "BrC[nHCl]",
            "Cl.Cc1(Br)ccC",
        ]

        correct_tokens = [
            ["^", "Br", "C", "[", "n", "H", "Cl", "]", "&"],
            ["^", "Cl", ".", "C", "c", "1", "(", "Br", ")", "c", "c", "C", "&"],
        ]

        tok = SMILESTokenizer(smiles=smiles, tokens=["Bq", "Br", "Cl"])

        smiles_tokenized = tok.tokenize(smiles)

        assert smiles_tokenized == correct_tokens

    def test_regex_tokens(self):
        smiles = [
            "NC[nHCl]",
            "C.CCCcc1(Br)cccC",
        ]

        correct_tokens = [
            ["^", "N", "C", "[nHCl]", "&"],
            ["^", "C", ".", "CCC", "cc", "1", "(", "B", "r", ")", "ccc", "C", "&"],
        ]

        tok = SMILESTokenizer(
            smiles=smiles, regex_token_patterns=[r"\[[^\]]+\]", "[c]+", "[C]+"]
        )

        smiles_tokenized = tok.tokenize(smiles)

        assert smiles_tokenized == correct_tokens

    # TODO Amiguity of Sc and Sn should be fixed
    def test_tokenize_detokenize_inverse(self, tokenizer, get_test_smiles):
        tokenized_data = tokenizer.tokenize(get_test_smiles)
        detokenized_data = tokenizer.detokenize(tokenized_data)

        assert detokenized_data == get_test_smiles

    def test_detokenize_new_lines_and_control_and_padding(self, tokenizer):
        smiles = ["^CN1&\n", "^cccCl&\n"]
        tokens = [
            [" ", "^", "C", "N", "1", "&"],
            ["^", "c", "c", "c", "Cl", "&", " "],
            ["^", "c", "c", "c", "Cl", "&", "Br", " "],
        ]

        smiles_raw = tokenizer.detokenize(tokens)
        smiles_control = tokenizer.detokenize(tokens, include_control_tokens=True)
        smiles_truncated = tokenizer.detokenize(
            tokens, include_control_tokens=False, truncate_at_end_token=True
        )

        smiles_end_of_line = tokenizer.detokenize(
            tokens, include_end_of_line_token=True
        )
        smiles_all = tokenizer.detokenize(
            tokens, include_end_of_line_token=True, include_control_tokens=True
        )

        for smi, smi_detokenized in zip(smiles, smiles_raw):
            assert smi[1:-2] == smi_detokenized

        for smi, smi_detokenized in zip(smiles, smiles_control):
            assert smi[:-1] == smi_detokenized

        for smi, smi_detokenized in zip(smiles, smiles_end_of_line):
            assert smi[1:-2] + smi[-1:] == smi_detokenized

        for smi, smi_detokenized in zip(smiles, smiles_all):
            assert smi == smi_detokenized

        for smi, smi_truncated in zip(smiles, smiles_end_of_line):
            assert smi[1:-2] + smi[-1:] == smi_truncated

    def test_ids_to_encoding_to_ids(self, tokenizer, get_test_smiles):
        encoding_ids = tokenizer(get_test_smiles)
        encoding_oh = tokenizer.convert_ids_to_encoding(
            encoding_ids, encoding_type="one hot"
        )
        decoding_ids = tokenizer.convert_encoding_to_ids(
            encoding_oh, encoding_type="one hot"
        )

        for encoded_id, decoded_id in zip(encoding_ids, decoding_ids):
            assert torch.equal(encoded_id, decoded_id)

    def test_encode_decode_encode_index(self, tokenizer, get_test_smiles):
        encoded_data = tokenizer(get_test_smiles)
        decoded_smiles = tokenizer.decode(encoded_data)

        for smi, smi_decoded in zip(get_test_smiles, decoded_smiles):
            assert smi == smi_decoded

    def test_encode_decode_encode_one_hot(self, tokenizer, get_test_smiles):
        encoded_data = tokenizer(get_test_smiles, encoding_type="one hot")
        decoded_smiles = tokenizer.decode(encoded_data, encoding_type="one hot")

        for smi, smi_decoded in zip(get_test_smiles, decoded_smiles):
            assert smi == smi_decoded

    def test_save_and_load(self, tokenizer, tmpdir):
        test_smiles = ["C.CCCcc1(Br)cccC"]
        filename = str(tmpdir / "vocab.json")

        tokenizer.save_vocabulary(filename)

        assert os.path.exists(filename)

        tokenizer2 = SMILESTokenizer(filename=filename)

        assert tokenizer(test_smiles)[0].tolist() == tokenizer2(test_smiles)[0].tolist()

        with pytest.warns(Warning):
            tokenizer3 = SMILESTokenizer()

        assert tokenizer(test_smiles)[0].tolist() != tokenizer3(test_smiles)[0].tolist()


class TestSMILESAtomTokenizer:
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
    def get_tokens(self):
        try:
            current_dir = os.path.dirname(__file__)
            # The file 'multi_char_atoms.txt' does not contain 'Sc' since it is
            # amiguous w.r.t. 'S' followed by 'c'.
            with open(os.path.join(current_dir, "multi_char_atoms.txt"), "r") as file:
                tokens = [t for t in file.read().split() if not t.startswith("#")]
        except FileNotFoundError:
            print("Cannot find 'multi_char_atoms.txt'")

        return tokens

    def test_default_atom_tokens(self):
        with pytest.warns(Warning):
            atom_tokenizer = SMILESAtomTokenizer()

        assert atom_tokenizer.vocabulary != {}
        assert atom_tokenizer.decoder_vocabulary != {}
        assert isinstance(atom_tokenizer.re, re.Pattern)

    def test_atom_tokens(self, get_test_smiles, get_tokens):
        tokenizer = SMILESTokenizer(smiles=get_test_smiles, tokens=get_tokens)

        atom_tokenizer = SMILESAtomTokenizer(smiles=get_test_smiles)

        assert tokenizer.vocabulary == atom_tokenizer.vocabulary

        for tokens, atom_tokens in zip(
            tokenizer(get_test_smiles), atom_tokenizer(get_test_smiles)
        ):

            assert torch.equal(tokens, atom_tokens)
