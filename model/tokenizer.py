# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# modified from https://github.com/facebookresearch/esm/blob/main/esm/data.py


from typing import Sequence, List
import itertools
from transformers import AutoTokenizer
import re

NUCLEOTIDES = ["A", "C", "G", "T"]
IUPAC_VALID = ["R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"]
EXTRA_NUCLEOTIDES = ["N", "-", "."]


proteinseq_toks = {
    "toks": [
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
    ]
}
dnaseq_toks = {
    "toks": [
        "G",
        "C",
        "A",
        "T",
        "N",
    ]
}


class ESM_Alphabet(object):
    """modified to pad up to multiples of 8"""

    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<bos>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        self.all_toks.extend(self.append_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i + 1}>")

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ["<eos>", "<unk>", "<pad>", "<cls>", "<mask>"]
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, truncation_seq_length: int = None):
        if self.use_msa:
            return MSABatchConverter(self, truncation_seq_length)
        else:
            return BatchConverter(self, truncation_seq_length)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("dna1"):  # placeholder
            standard_toks = dnaseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(
            standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa
        )

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


class NucleotideToken(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        special_toks: Sequence[str] = (
            "<chr_cls>",
            "<chr_bos>",
            "<chr_eos>",
            "<mask>",
            "<unk>",
            "<null_0>",
            "<pad>",
            "<crop_separator>",
            "<concat_separator>",
        ),
    ):
        self.standard_toks = list(standard_toks)
        self.special_toks = list(special_toks)
        self.all_toks = self.standard_toks + self.special_toks
        for i in range((16 - (len(self.all_toks) % 16)) % 16):
            self.all_toks.append(f"<null_{i + 1}>")

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.tok_to_idx["<pad>"]
        self.mask_idx = self.tok_to_idx["<mask>"]
        self.cls_idx = self.tok_to_idx["<chr_cls>"]
        self.bos_idx = self.tok_to_idx["<chr_bos>"]
        self.eos_idx = self.tok_to_idx["<chr_eos>"]
        self.null_idx = self.tok_to_idx["<null_0>"]
        self.crop_separator_idx = self.tok_to_idx["<crop_separator>"]
        self.concat_separator_idx = self.tok_to_idx["<concat_separator>"]

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    # def encode(self, text):
    #     return [self.tok_to_idx[tok] for tok in [*text]]

    def encode(self, text):
        tokens = []
        i = 0
        length = len(text)

        while i < length:
            if (
                text[i] == "<"
            ):  # Check if the current character indicates the start of a special token
                special_start = i
                while (
                    i < length and text[i] != ">"
                ):  # Find the end of the special token
                    i += 1
                if (
                    i < length and text[i] == ">"
                ):  # Check for the end of a special token
                    special = text[special_start : i + 1]
                    try:
                        special_idx = self.tok_to_idx[special]
                        tokens.append(special_idx)
                    except KeyError:
                        raise ValueError(f"Invalid special token: {special}")
                else:
                    raise ValueError(
                        f"Incomplete special token: {text[special_start:]}"
                    )
            else:
                tokens.append(self.tok_to_idx[text[i]])
            i += 1

        return tokens

    def encode_batch(
        self,
        texts,
        crop_separator: List[bool],
        concat_separator: List[bool],
        bos_eos: bool = False,
        cls: bool = False,
    ):
        """Encode a batch of texts into a list of token indices"""
        encoded_output = []
        for i, text in enumerate(texts):
            tokens = self.encode(text)
            if bos_eos:
                tokens = [self.bos_idx] + tokens + [self.eos_idx]
            if cls:
                tokens = [self.cls_idx] + tokens
            if i < len(texts) - 1:
                if crop_separator:
                    tokens = tokens + [self.crop_separator_idx]
                if concat_separator:
                    tokens = tokens + [self.concat_separator_idx]
            encoded_output.append(tokens)
        return encoded_output

    @classmethod
    def from_architecture(cls, name: str) -> "Tokenizer":
        if name in ("dna1"):  # placeholder
            standard_toks = list(NUCLEOTIDES + IUPAC_VALID + EXTRA_NUCLEOTIDES)
            special_toks = (
                "<chr_cls>",
                "<chr_bos>",
                "<chr_eos>",
                "<mask>",
                "<unk>",
                "<null_0>",
                "<pad>",
                "<crop_separator>",
                "<concat_separator>",
            )
        elif name in ("dna-large"):  # Larger vocab size padded up to 128 dim.
            standard_toks = list(NUCLEOTIDES + IUPAC_VALID + EXTRA_NUCLEOTIDES) + list(
                range(96)
            )
            special_toks = (
                "<chr_cls>",
                "<chr_bos>",
                "<chr_eos>",
                "<mask>",
                "<unk>",
                "<null_0>",
                "<pad>",
                "<crop_separator>",
                "<concat_separator>",
            )
        elif "dna-" in name:
            match = re.search(r"dna-(\d+)", name)
            vocab_size = int(match.group(1)) # Should return ValueError if format incorrect
            standard_toks = list(NUCLEOTIDES + IUPAC_VALID + EXTRA_NUCLEOTIDES) + list(
                range(vocab_size-32)
            )
            special_toks = (
                "<chr_cls>",
                "<chr_bos>",
                "<chr_eos>",
                "<mask>",
                "<unk>",
                "<null_0>",
                "<pad>",
                "<crop_separator>",
                "<concat_separator>",
            )
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, special_toks)


class NucleotideTransformerTokenizer(object):
    def __init__(self, *args, **kargs):
        """tokenizer.model_max_length == 2048"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            trust_remote_code=True,
        )
        self.tok_to_idx = self.tokenizer.get_vocab()
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.tok_to_idx["<pad>"]
        self.mask_idx = self.tok_to_idx["<mask>"]
        self.cls_idx = self.tok_to_idx["<cls>"]
        self.bos_idx = self.tok_to_idx["<bos>"]
        self.eos_idx = self.tok_to_idx["<eos>"]

    def __len__(self):
        return len(self.tokenizer)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.tokenizer.all_tokens[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in [*text]]

    def encode_batch(self, texts, bos_eos=False, cls=False):
        # truncation=True was not their default
        return self.tokenizer.batch_encode_plus(
            texts,
            return_tensors=None,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )["input_ids"]


tokenizer_registry = {
    "ESM_Alphabet": ESM_Alphabet,
    "NucleotideToken": NucleotideToken,
    "NucleotideTransformerTokenizer": NucleotideTransformerTokenizer,
}
