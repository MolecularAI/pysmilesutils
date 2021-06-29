"""Tools for analyzis of the distribution of tokens in the SMILES dataset
"""
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from pysmilesutils.tokenize import SMILESTokenizer


def analyze_smiles_tokens(
    tokenizer: SMILESTokenizer, smiles: List[str]
) -> Dict[str, Any]:
    """Analyzes a list of SMILES and returns statistics in the form of a dictionary.

    This function applies the `tokenize` function of the specified tokenizer and
    extracts a set of statistics for the supplied SMILES.
    These include:
      - token frequency,
      - average number of tokens, and
      - average number of characters.

    :param tokenizer: A tokenizer to be used for token analysis.
    :param smiles: A list of smiles to be analyzed.
    :return: A dictionary containing a number of different SMILES statistics.
    """

    token_frequency_d: Dict[str, int] = defaultdict(int)
    num_tokens_d: Dict[int, int] = defaultdict(int)
    num_characters_d: Dict[int, int] = defaultdict(int)

    for smi, tokens in zip(smiles, tokenizer.tokenize(smiles)):
        num_tokens_d[len(tokens)] += 1
        num_characters_d[len(smi)] += 1
        for token in tokens:
            token_frequency_d[token] += 1

    num_tokens: Tuple[Tuple[Any, ...], ...] = tuple(zip(*sorted(num_tokens_d.items())))
    token_frequency: Tuple[Tuple[Any, ...], ...] = tuple(zip(*sorted(token_frequency_d.items())))
    avg_num_tokens: float = sum(token_frequency[1]) / len(smiles)
    max_num_tokens: int = max(map(len, smiles))
    avg_num_characters: float = sum(map(len, smiles)) / len(smiles)
    max_num_characters: int = max(map(len, smiles))

    return {
        "token_frequency": token_frequency,
        "num_tokens": num_tokens,
        "avg_num_tokens": avg_num_tokens,
        "max_num_tokens": max_num_tokens,
        "num_characters": num_tokens,
        "avg_num_characters": avg_num_characters,
        "max_num_characters": max_num_characters,
    }
