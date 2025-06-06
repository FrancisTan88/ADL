from typing import Iterable, List

# 把batch裡的每個資料丟進來encode: to idx
class Vocab:
    PAD = "[PAD]"   # padding 
    UNK = "[UNK]"   # unknown 

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]  # return 0

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]  # return 1

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)  # return idx if token exists, else return 1(unknown)

    def encode(self, tokens: List[str]) -> List[int]:  # tokens 
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens] # convert to 'list of index', batch_ids is a 2D array
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len  
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id) # padding
        return padded_ids


# 將每個list of index長度不足的部分補0
def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds
