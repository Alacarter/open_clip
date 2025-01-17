import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
from nltk.lm import Vocabulary


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe(), special_tokens=None):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        if not special_tokens:
            special_tokens = ['<start_of_text>', '<end_of_text>']
        else:
            special_tokens = ['<start_of_text>', '<end_of_text>'] + special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t:t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


class Vocab(Vocabulary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.tokens = ['<PAD>', '<UNK>', '<start_of_text>', '<end_of_text>'] + list(self.counts.keys())

    def tokens_to_idx(self, text, output_list=True):
        tokens = text.split(" ")
        if len(tokens) == 1 and not output_list:
            token = tokens[0]
            if token in self.tokens:
                return self.tokens.index(token)
            else:
                return self.tokens.index('<UNK>')
        else:
            idxs = []
            for token in tokens:
                idx = self.tokens_to_idx(token, output_list=False)
                idxs.append(idx)
            return idxs

    def idxs_to_tokens(self, idxs):
        assert isinstance(tokens, list)
        return [self.tokens[idx] for idx in idxs]

    def size(self):
        return len(self.tokens)


class CustomTokenizer(object):
    def __init__(self):
        self.vocab = Vocab([], unk_cutoff=1)
        vocab_path = "/scratch/cluster/albertyu/datasets/202110260345_fruitbot_image_instruction/20211026_train_20211220_vocab.txt"
        # vocab_path = "/scratch/cluster/albertyu/datasets/20220126_fruitbot_image_instruction/20220126_train_20211220_vocab.txt"
        with open(vocab_path, "r") as f:
            vocab_list = f.readlines()
        vocab_list = [basic_clean(vocab) for vocab in vocab_list]
        self.vocab.update(vocab_list)
        self.sot_token = self.vocab.tokens_to_idx('<start_of_text>')[0]
        self.eot_token = self.vocab.tokens_to_idx('<end_of_text>')[0]

    def encode(self, text):
        return self.vocab.tokens_to_idx(text)

    def decode(self, tokens):
        return self.vocab.idxs_to_tokens(tokens)


# if __name__ == "__main__":
#     tok = SimpleTokenizer()
#     tok = CustomTokenizer()
#     import ipdb; ipdb.set_trace()
#     print(tok.encode("banana apple wall afdsaf"))
#     # print(tok.encoder)
#     # print("sot", sot_token, "eot", eot_token)
