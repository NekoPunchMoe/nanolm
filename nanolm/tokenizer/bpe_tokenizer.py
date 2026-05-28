import re
import regex
from collections import defaultdict
from collections.abc import Iterable, Iterator

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = []):
        self.vocab = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def _get_pairs(self, splits: list[bytes]) -> list[tuple[bytes, bytes]]:
        return [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]

    def _pre_tokenize(self, text: str,) -> list[list[bytes]]:
        pre_tokens = []
        for pre_token in regex.finditer(self.PAT, text):
            pre_token_text = pre_token.group(0)
            cur_token_bytes = [bytes([b]) for b in pre_token_text.encode("utf-8")]
            pre_tokens.append(cur_token_bytes)
        return pre_tokens

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            # CRITICAL: Sort special tokens by length in descending order (longest first).
            # Reason: Regex OR (|) evaluates left-to-right and stops on the first match. 
            # Sorting prevents shorter tokens from prematurely splitting longer, overlapping ones.
            # 
            # Example: 
            # Tokens: ["<|end|>", "<|endoftext|>"]
            # Text:   "hello <|endoftext|>"
            # - Unsorted regex: "(<|end|>|<|endoftext|>)" -> Matches "<|end|>", ruining the longer token (leaves "oftext|>").
            # - Sorted regex:   "(<|endoftext|>|<|end|>)" -> Correctly matches the full "<|endoftext|>".
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escape_special_tokens = [regex.escape(t) for t in sorted_special_tokens]
            # IMPORTANT: unlike bpe tokenizer training process, we want to make sure special tokens are kept as whole units during encoding
            # so we add () in pattern to capture them as separate tokens during splitting.
            split_text = regex.split(f"({'|'.join(escape_special_tokens)})", text)
        else:
            split_text = [text]
        token_ids = []

        for cur_text in split_text:
            if not cur_text:
                continue

            if cur_text in self.special_tokens:
                # IMPORTANT: when wen find a special token, we directly convert it to token id without further pre-tokenization or merging
                # DO NOT put this in pre-tokenization or merging process, otherwise the special token might be split into pieces and lose its meaning.
                token_ids.append(self.token_to_id[cur_text.encode("utf-8")])
                continue

            pre_tokens = self._pre_tokenize(cur_text)
            pairs_to_idices = defaultdict(set)
            for idx, pre_token in enumerate(pre_tokens):
                for pair in self._get_pairs(pre_token):
                    pairs_to_idices[pair].add(idx)
            
            for merge in self.merges:
                target_indices = list(pairs_to_idices.get(merge, set()))
                for idx in target_indices:
                    # remove old pairs that involve the tokens we're merging
                    for pair in self._get_pairs(pre_tokens[idx]):
                        pairs_to_idices[pair].discard(idx)
                    
                    # create new token for the merged pair
                    new_splits = []
                    i = 0
                    while i < len(pre_tokens[idx]):
                        if i < len(pre_tokens[idx]) - 1 and (pre_tokens[idx][i], pre_tokens[idx][i + 1]) == merge:
                            new_splits.append(merge[0] + merge[1])
                            i += 2
                        else:
                            new_splits.append(pre_tokens[idx][i])
                            i += 1
                    pre_tokens[idx] = new_splits

                    # add new pairs that involve the new merged token
                    for pair in self._get_pairs(pre_tokens[idx]):
                        pairs_to_idices[pair].add(idx)
            # Flatten the list of pre_tokens and convert to token ids
            for pre_token in pre_tokens:
                for token in pre_token:
                    token_ids.append(self.token_to_id[token])
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        bytes_list = [self.vocab[token_id] for token_id in token_ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")
        