"""SMILES Tokenizer module.
"""
import re
import json
import warnings
from re import Pattern
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Any

import torch

Tokens = List[str]


class SMILESTokenizer:
    """A class for tokenizing and encoding SMILES.

    The tokenizer has a vocabulary that maps tokens to unique integers (a dictionary Dict[str, int]),
    and is created from a set of SMILES. Unless specified otherwise all single character are treated as tokens,
    but the user can specify additional tokens with a list of strings, as well as a list of regular expressions.
    Using the tokenized SMILES the tokenizer also encodes data to lists of `torch.Tensor`.

    This class can also be extended to allow for different and more advanced tokenization schemes.
    When extending the class the functions `tokenize`, `convert_tokens_to_ids`, and `convert_ids_to_encodings`
    can be overriden to change the tokenizers behaviour. These three function are all used in the `encode` function
    which constitutes the entire tokenization pipeline from SMILES to encodings. When modifying the the three
    aformentioned functions the inverses should be modified if necessary,
    these are: `convert_encoding_to_ids`, `convert_ids_to_tokens`, and `detokenize`,
    and are used in the `decode` function.

    Calling an instance of the class on a list of SMILES (or a single SMILES) will return a  list of torch tensors
    with the encoded data, and is equivalent to calling `encode`.

    Inspiration for this tokenizer Class was taken from https://huggingface.co/transformers/main_classes/tokenizer.html
    and https://github.com/MolecularAI/Reinvent/blob/master/models/vocabulary.py

    Initializes the SMILESTokenizer by setting necessary parameters as well as
    compiling regular expressions form the given token, and regex_token lists.
    If a list of SMILES is provided a vocabulary is also created using this list.

    Note that both the token and regex list are used when creating the vocabulary of the tokenizer.
    Note also that the list of regular expressions take priority when parsing the SMILES,
    and tokens earlier are in the lists are also prioritized.

    The `encoding_type` argument specifies the type of encoding used. Must be either 'index' or 'one hot'.
    The former means that the encoded data are integer representations of the tokens found,
    while the latter is one hot encodings of these ids. Defaults to "index".

    :param smiles: A list of SMILES that are used to create the vocabulary for the tokenizer. Defaults to None.
    :param tokens:  A list of tokens (strings) that the tokenizer uses when tokenizing SMILES. Defaults to None.
    :param regex_token_patterns: A list of regular expressions that the tokenizer uses when tokenizing SMILES.
    :param beginning_of_smiles_token: Token that is added to beginning of SMILES. Defaults to "^".
    :param end_of_smiles_token: Token that is added to the end of SMILES. Defaults to "&".
    :param padding_token: Token used for padding. Defalts to " ".
    :param unknown_token: Token that is used for unknown ids when decoding encoded data. Defaults to "?".
    :param encoding_type: The type of encoding used for the final output.
    :param filename: if given and `smiles` is None, load the vocabulary from disc
    :raises: ValueError: If the `encoding_type` is invalid.
    """

    def __init__(
        self,
        smiles: List[str] = None,
        tokens: List[str] = None,
        regex_token_patterns: List[str] = None,
        beginning_of_smiles_token: str = "^",
        end_of_smiles_token: str = "&",
        padding_token: str = " ",
        unknown_token: str = "?",
        encoding_type: str = "index",  # "one hot" or "index"
        filename: str = None,
    ) -> None:
        self._check_encoding_type(encoding_type)

        self.encoding_type = encoding_type
        self._beginning_of_smiles_token = beginning_of_smiles_token
        self._end_of_smiles_token = end_of_smiles_token
        self._padding_token = padding_token
        self._unknown_token = unknown_token

        self._regex_tokens: List[str] = []
        self._tokens: List[str] = []

        smiles = smiles or []
        regex_token_patterns = regex_token_patterns or []
        tokens = tokens or []

        with warnings.catch_warnings(record=smiles != [] or filename):
            self.add_regex_token_patterns(regex_token_patterns)
            self.add_tokens(tokens)

        self._re: Optional[Pattern] = None
        self._vocabulary: Dict[str, int] = {}
        self._decoder_vocabulary: Dict[int, str] = {}
        if smiles:
            self.create_vocabulary_from_smiles(smiles)
        elif filename:
            self.load_vocabulary(filename)

    @property
    def special_tokens(self) -> Dict[str, str]:
        """Returns a dictionary of non-character tokens"""
        return {
            "start": self._beginning_of_smiles_token,
            "end": self._end_of_smiles_token,
            "pad": self._padding_token,
            "unknown": self._unknown_token,
        }

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Tokens vocabulary.

        :return: Tokens vocabulary
        """
        if not self._vocabulary:
            self._vocabulary = self._reset_vocabulary()
        return self._vocabulary

    @property
    def decoder_vocabulary(self) -> Dict[int, str]:
        """Decoder tokens vocabulary.

        :return: Decoder tokens vocabulary
        """
        if not self._decoder_vocabulary:
            self._decoder_vocabulary = self._reset_decoder_vocabulary()
        return self._decoder_vocabulary

    @property
    def re(self) -> Pattern:
        """Tokens Regex Object.

        :return: Tokens Regex Object
        """
        if not self._re:
            self._re = self._get_compiled_regex(self._tokens, self._regex_tokens)
        return self._re

    def __call__(
        self, data: Union[str, List[str]], *args, **kwargs
    ) -> List[torch.Tensor]:
        return self.encode(data, *args, **kwargs)

    def __len__(self) -> int:
        return len(self.vocabulary)

    def __getitem__(self, item: str) -> int:
        if item in self.special_tokens:
            return self.vocabulary[self.special_tokens[item]]
        if item not in self.vocabulary:
            raise KeyError(f"Unknown token: {item}")
        return self.vocabulary[item]

    def _reset_vocabulary(self) -> Dict[str, int]:
        """Create a new tokens vocabulary.

        :return: New tokens vocabulary
        """
        return {
            self._padding_token: 0,
            self._beginning_of_smiles_token: 1,
            self._end_of_smiles_token: 2,
            self._unknown_token: 3,
        }

    def _reset_decoder_vocabulary(self) -> Dict[int, str]:
        """Create a new decoder tokens vocabulary.

        :return: New decoder tokens vocabulary
        """
        return {i: t for t, i in self.vocabulary.items()}

    def encode(
        self,
        data: Union[List[str], str],
        encoding_type: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """Encodes a list of SMILES or a single SMILES into torch tensor(s).

        The encoding is specified by the tokens and regex supplied to the tokenizer
        class. This function uses the three functions `tokenize`,
        `convert_tokens_to_ids`, and `convert_ids_to_encodings` as the encoding
        process.

        :param data: A list of SMILES or a single SMILES.
        :param encoding_type: The type of encoding to convert to,
                'index' or 'one hot'. If `None` is provided the value specified in
                the class is used., defaults to None

        :raises ValueError: If the `encoding_type` is invalid.

        :return:  A list of tensors containing the encoded SMILES.
        """
        if encoding_type is None:
            encoding_type = self.encoding_type

        self._check_encoding_type(encoding_type)
        if isinstance(data, str):
            # Convert string to a list with one string
            data = [data]

        tokenized_data = self.tokenize(data)
        id_data = self.convert_tokens_to_ids(tokenized_data)
        encoded_data = self.convert_ids_to_encoding(id_data, encoding_type)

        return encoded_data

    def tokenize(self, data: List[str]) -> List[List[str]]:
        """Tokenizes a list of SMILES into lists of tokens.

        The conversion is done by parsing the SMILES using regular expressions, which have been
        compiled using the token and regex lists specified in the tokenizer. This
        function is part of the SMILES encoding process and is called in the
        `encode` function.

        :param data: A list os SMILES to be tokenized.

        :return: Lists of tokens.
        """
        tokenized_data = []

        for smi in data:
            tokens = self.re.findall(smi)
            tokenized_data.append(
                [self._beginning_of_smiles_token] + tokens + [self._end_of_smiles_token]
            )

        return tokenized_data

    def convert_tokens_to_ids(self, token_data: List[List[str]]) -> List[torch.Tensor]:
        """Converts lists of tokens to lists of token ids.

        The tokens are converted to ids using the tokenizers vocabulary.

        :param token_data: Lists of tokens to be converted.

        :return: Lists of token ids that have been converted from tokens.
        """
        tokens_lengths = list(map(len, token_data))
        ids_list = []

        for tokens, length in zip(token_data, tokens_lengths):
            ids_tensor = torch.zeros(length, dtype=torch.long)
            for tdx, token in enumerate(tokens):
                ids_tensor[tdx] = self.vocabulary.get(
                    token, self.vocabulary[self._unknown_token]
                )
            ids_list.append(ids_tensor)

        return ids_list

    def convert_ids_to_encoding(
        self, id_data: List[torch.Tensor], encoding_type: Optional[str] = None
    ) -> List[torch.Tensor]:
        """Converts a list of id tensors to a list of tensors of SMILES encodings.

        The function is used when encoding SMILES and is called in the `encode`
        function. If the `encoding_type` is `index` then the input is returned.

        :param id_data: A list of tensors containing
                token ids.
        :param encoding_type: The type of encoding to convert to,
                'index' or 'one hot'. If `None` is provided the value specified in
                the class is used., defaults to None

        :raises ValueError: If the `encoding_type` is invalid.

        :return: List of tensors of encoded SMILES.
        """
        if encoding_type is None:
            encoding_type = self.encoding_type

        self._check_encoding_type(encoding_type)

        if encoding_type == "index":
            return id_data
        # Implies "one hot" encoding
        num_tokens = len(self.vocabulary)
        onehot_tensor = torch.eye(num_tokens)
        onehot_data = [onehot_tensor[ids] for ids in id_data]
        return onehot_data

    def decode(
        self, encoded_data: List[torch.Tensor], encoding_type: Optional[str] = None
    ) -> List[str]:
        """Decodes a list of SMILES encodings back into SMILES.

        This function is the inverse of `encode` and utilizes the three functions
        `convert_encoding_to_ids`, `convert_ids_to_tokens`, and `detokenize`.

        :param encoded_data: The encoded SMILES data to be
                decoded into SMILES.
        :param encoding_type: The type of encoding to convert from,
                'index' or 'one hot'. If `None` is provided the value specified in
                the class is used., defaults to None

        :return: A list of SMILES.
        """
        id_data = self.convert_encoding_to_ids(encoded_data, encoding_type)
        tokenized_data = self.convert_ids_to_tokens(id_data)
        smiles = self.detokenize(tokenized_data)

        return smiles

    def detokenize(
        self,
        token_data: List[List[str]],
        include_control_tokens: bool = False,
        include_end_of_line_token: bool = False,
        truncate_at_end_token: bool = False,
    ) -> List[str]:
        """Detokenizes lists of tokens into SMILES by concatenating the token strings.

        This function is used in the `decode` function when decoding
        data into SMILES, and it is the inverse of `tokenize`.

        :param token_data: Lists of tokens to be detokenized.
        :param include_control_tokens: If `False` the beginning
            and end tokens are stripped from the token lists. Defaults to False
        :param include_end_of_line_token: If `True` end of line
            characters `\\n` are added to the detokenized SMILES. Defaults to False
        :param truncate_at_end_token: If `True`, all tokens after the end-token is removed.
            Defaults to False.

        :return: A list of detokenized SMILES.
        """

        character_lists = [tokens.copy() for tokens in token_data]

        character_lists = [
            self._strip_list(
                tokens,
                strip_control_tokens=not include_control_tokens,
                truncate_at_end_token=truncate_at_end_token,
            )
            for tokens in character_lists
        ]

        if include_end_of_line_token:
            for s in character_lists:
                s.append("\n")

        strings = ["".join(s) for s in character_lists]

        return strings

    def convert_ids_to_tokens(self, ids_list: List[torch.Tensor]) -> List[List[str]]:
        """Converts lists of token ids to a token tensors.

        This function is used when decoding data using the `decode` function,
        and is the inverse of `convert_tokens_to_ids`.

        :param ids_list: A list of Tensors where each
                Tensor containts the ids of the tokens it represents.

        :return: A list where each element is a list of the
                tokens corresponding to the input ids.
        """
        tokens_data = []
        for ids in ids_list:
            tokens = [self.decoder_vocabulary[i] for i in ids.tolist()]
            tokens_data.append(tokens)

        return tokens_data

    def convert_encoding_to_ids(
        self, encoded_data: List[torch.Tensor], encoding_type: Optional[str] = None
    ) -> List[torch.Tensor]:
        """Converts a list of encodings of SMILES to a list of id tensors.

        This functions is used when decoding data with the `decode` function,
        and is the inverse of `covert_ids_to_encoding`. If the encoding type is
        'index' this function just returns the input.

        :param encoded_data: Encoded SMILES to be
                converted.
        :param encoding_type: The type of encoding to convert from,
                'index' or 'one hot'. If `None` is provided the value specified in
                the class is used., defaults to None

        :raises ValueError: If the `encoding_type` is invalid.

        :return: A list of tensors containing the token ids.
        """
        if encoding_type is None:
            encoding_type = self.encoding_type

        self._check_encoding_type(encoding_type)

        if encoding_type == "index":
            return encoded_data

        # Implies "one hot" encoding
        id_data = []
        for encoding in encoded_data:
            indices, t_ids = torch.nonzero(encoding, as_tuple=True)
            ids = torch.zeros(encoding.shape[0], dtype=torch.long)
            ids[indices] = t_ids
            id_data.append(ids)

        return id_data

    def add_tokens(self, tokens: List[str], regex: bool = False, smiles=None) -> None:
        """Adds tokens to the classes list of tokens.

        The new tokens are added to the front of the token list and take priority over old tokens. Note that that the
        vocabulary of the tokenizer is not updated after the tokens are added,
        and must be updated by calling `create_vocabulary_from_smiles`.

        :param tokens: List of tokens to be added.
        :param regex: If `True` the input tokens are treated as
                regular expressions and are added to the list of regular expressions
                instead of token list. Defaults to False.
        :param smiles: If a list of smiles is provided, the vocabulary will be created, defaults to None

        :raises ValueError: If any of the tokens supplied are already in the list
                of tokens.
        """
        existing_tokens = self._regex_tokens if regex else self._tokens
        for token in tokens:
            if token in existing_tokens:
                raise ValueError(f'"{token}" already present in list of tokens.')

        if regex:
            self._regex_tokens[0:0] = tokens
        else:
            self._tokens[0:0] = tokens

        # Get a compiled tokens regex
        self._re = self._get_compiled_regex(self._tokens, self._regex_tokens)

        if not smiles:
            warnings.warn(
                "Tokenizer vocabulary has not been updated. Call `create_vocabulary_from_smiles`\
                with SMILES data to update."
            )
        else:
            self.create_vocabulary_from_smiles(smiles)

    def add_regex_token_patterns(
        self, tokens: List[str], smiles: List[str] = None
    ) -> None:
        """Adds regular expressions to the list used when tokenizing SMILES.

        This function is a shorthand for `add_tokens(tokens, regex=True)`.

        :param tokens: A list of regular expressions.
        :param smiles: If a list of smiles are provided, the vocabulary will be created using these, defaults to None
        """
        self.add_tokens(tokens, regex=True, smiles=smiles)

    def create_vocabulary_from_smiles(self, smiles: List[str]) -> None:
        """Creates a vocabulary by iteratively tokenizing the SMILES and adding
        the found tokens to the vocabulary.

        A `vocabulary` is a dictionary that maps tokens (str) to integers.
        The tokens vocabulary is not the same as the list of tokens,
        since tokens are also found by applying the list of regular expressions.

        A `decoder_vocabulary` is the inverse of the
        vocabulary. It is always possible to create an inverse since the vocabulary
        always maps to unique integers.


        :param smiles: List of SMILES whose tokens are used to create
                the vocabulary.
        """
        # Reset Tokens Vocabulary
        self._vocabulary = self._reset_vocabulary()

        for tokens in self.tokenize(smiles):
            for token in tokens:
                self._vocabulary.setdefault(token, len(self._vocabulary))

        # Reset decoder tokens vocabulary
        self._decoder_vocabulary = self._reset_decoder_vocabulary()

    def remove_token_from_vocabulary(self, token: str) -> None:
        """Removes a token from the tokenizers `vocabulary` and the corresponding
        entry in the `decoder_vocabulary`.

        :param token: Token to be removed from `vocabulary`.

        :raises ValueError: If the specified token can't be found on the `vocabulary`.
        """
        vocabulary_tokens: List[str] = list(self.vocabulary.keys())

        if token not in vocabulary_tokens:
            raise ValueError(f"{token} is not in the vocabulary")

        vocabulary_tokens.remove(token)

        # Recreate tokens vocabulary
        self._vocabulary = {t: i for i, t in enumerate(vocabulary_tokens)}

    def load_vocabulary(self, filename: str) -> None:
        """
        Load a serialized vocabulary from a JSON format

        :param filename: the path to the file on disc
        """
        with open(filename, "r") as fileobj:
            dict_ = json.load(fileobj)

        self._update_state(dict_["properties"])
        self._vocabulary = {token: idx for idx, token in enumerate(dict_["vocabulary"])}
        self._reset_decoder_vocabulary()

    def save_vocabulary(self, filename: str) -> None:
        """
        Save the vocabulary to a JSON format.

        :param filename: the path to the file on disc
        """
        token_tuples = sorted(self.vocabulary.items(), key=lambda k_v: k_v[1])
        tokens = [key for key, _ in token_tuples]
        dict_ = {"properties": self._state_properties(), "vocabulary": tokens}
        with open(filename, "w") as fileobj:
            json.dump(dict_, fileobj, indent=4)

    def _strip_list(
        self,
        tokens: List[str],
        strip_control_tokens: bool = False,
        truncate_at_end_token: bool = False,
    ) -> List[str]:
        """Cleanup tokens list from control tokens.

        :param tokens: List of tokens
        :param strip_control_tokens: Flag to remove control tokens, defaults to False
        :param truncate_at_end_token: If True truncate tokens after end-token
        """
        if truncate_at_end_token and self._end_of_smiles_token in tokens:
            end_token_idx = tokens.index(self._end_of_smiles_token)
            tokens = tokens[: end_token_idx + 1]

        strip_characters: List[str] = [self._padding_token]
        if strip_control_tokens:
            strip_characters.extend(
                [self._beginning_of_smiles_token, self._end_of_smiles_token]
            )
        while len(tokens) > 0 and tokens[0] in strip_characters:
            tokens.pop(0)

        while len(tokens) > 0 and tokens[-1] in strip_characters:
            tokens.pop()

        return tokens

    def _get_compiled_regex(
        self, tokens: List[str], regex_tokens: List[str]
    ) -> Pattern:
        """Create a Regular Expression Object from a list of tokens and regular expression tokens.

        :param tokens: List of tokens
        :param regex_tokens: List of regular expression tokens
        :return: Regular Expression Object
        """
        regex_string = r"("
        for token in tokens:
            processed_token = token
            for special_character in "()[]":
                processed_token = processed_token.replace(
                    special_character, f"\\{special_character}"
                )
            regex_string += processed_token + r"|"
        for token in regex_tokens:
            regex_string += token + r"|"
        regex_string += r".)"

        return re.compile(regex_string)

    def _check_encoding_type(self, encoding_type: str) -> None:
        """Check if encoding type is one of "index" or "one hot".

        :param encoding_type: Encoding type
        :raises ValueError: If encoding_type is not one of "index" or "one hot"
        """
        if encoding_type not in {"one hot", "index"}:
            raise ValueError(
                f"unknown choice of encoding: {encoding_type}, muse be either 'one hot' or 'index'"
            )

    def _state_properties(self) -> Dict[str, Any]:
        """Return properties to reconstruct the internal state of the tokenizer"""
        dict_ = {"regex": self._re.pattern if self._re else ""}
        dict_["special_tokens"] = {
            name: val for name, val in self.special_tokens.items()
        }
        return dict_

    def _update_state(self, dict_: Dict[str, Any]) -> None:
        """Update the internal state with properties loaded from disc"""
        if dict_["regex"]:
            self._re = re.compile(dict_["regex"])
        else:
            self._re = None
        self._beginning_of_smiles_token = dict_["special_tokens"]["start"]
        self._end_of_smiles_token = dict_["special_tokens"]["end"]
        self._padding_token = dict_["special_tokens"]["pad"]
        self._unknown_token = dict_["special_tokens"]["unknown"]
        self._regex_tokens = []
        self._tokens = []


class SMILESAtomTokenizer(SMILESTokenizer):
    """A subclass of the `SMILESTokenizer` that treats all atoms as tokens.

    This tokenizer works by applying two different sets of regular expressions:
    one for atoms inside blocks ([]) and another for all other cases. This allows
    the tokenizer to find all atoms as blocks without having a comprehensive list
    of all atoms in the token list.
    """

    def __init__(
        self,
        *args,
        tokens: List[str] = None,
        smiles: List[str] = None,
        regex_tokens_patterns: List[str] = None,
        **kwargs,
    ) -> None:
        regex_tokens_patterns = regex_tokens_patterns or []

        smiles = smiles or []

        with warnings.catch_warnings(record=smiles != []):
            super().__init__(*args, **kwargs)
            super().add_tokens(["Br", "Cl"])
            super().add_regex_token_patterns(regex_tokens_patterns + [r"\[[^\]]*\]"])
        self.re_block_atom = re.compile(r"(Zn|Sn|Sc|[A-Z][a-z]?(?<!c|n|o|p|s)|se|as|.)")

        super().create_vocabulary_from_smiles(smiles)

    def tokenize(self, smiles: List[str]) -> List[List[str]]:
        """Converts a list of SMILES into a list of lists of tokens, where all atoms are
        considered to be tokens.

        The function first scans the SMILES for atoms and bracketed expressions
        uisng regular expressions. These bracketed expressions are then parsed
        again using a different regular expression.


        :param smiles: List of SMILES.

        :return: List of tokenized SMILES.
        """
        data_tokenized = super().tokenize(smiles)
        final_data = []
        for tokens in data_tokenized:
            final_tokens = []
            for token in tokens:
                if token.startswith("["):
                    final_tokens += self.re_block_atom.findall(token)
                else:
                    final_tokens.append(token)
            final_data.append(final_tokens)

        return final_data
