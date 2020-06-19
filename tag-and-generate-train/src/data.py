from collections import defaultdict
import torch as th
from torch.utils import data
import json
import numpy as np


def loadtxt(filename):
    txt = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            txt.append(line.rstrip())
    return txt


class Vocab(object):
    """Maps symbols (word/tokens) to indices"""

    def __init__(self):
        # Containers
        self.symbols = []
        self.idxs = {}
        # State
        self.frozen = False
        # Special symbols
        self.add_symbol("<pad>")  # Padding token
        self.add_symbol("<sos>")  # Start of sentence token
        self.add_symbol("<eos>")  # End of sentence token
        self.add_symbol("<unk>")  # Unknown token
        self.add_symbol("[GMASK]") # add GMASK

    def __len__(self):
        return len(self.idxs)

    def add_symbol(self, symbol):
        """Add a symbol to the dictionary and return its index

        If the symbol already exists in the dictionary this just returns
        the index"""
        if symbol not in self.idxs:
            if self.frozen:
                raise ValueError("Can't add symbol to frozen dictionary")
            self.symbols.append(symbol)
            # print(symbol, len(self.idxs))
            self.idxs[symbol] = len(self.idxs)
        return self.idxs[symbol]

    def to_idx(self, symbol):
        """Return symbol's index

        If the symbol is not in the dictionary, returns the index of <unk>"""
        if symbol in self.idxs:
            return self.idxs[symbol]
        else:
            return self.idxs["<unk>"]

    def to_symbol(self, idx):
        """Return idx's symbol"""
        return self.symbols[idx]

    def __getitem__(self, symbol_or_idx):
        if isinstance(symbol_or_idx, int):
            return self.to_symbol(symbol_or_idx)
        else:
            return self.to_idx(symbol_or_idx)

    @staticmethod
    def from_data_files(*filenames, max_size=-1, min_freq=2):  # AB Change 1
        """Builds a dictionary from the most frequent tokens in files"""
        vocab = Vocab()
        # Record token counts
        token_counts = defaultdict(lambda: 0)
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    tokens = line.rstrip().split()
                    for token in tokens:
                        token_counts[token] += 1
        # Filter out least frequent tokens
        token_counts = {
            tok: cnt
            for tok, cnt in token_counts.items()
            if cnt >= min_freq
        }
        # Only keep most common tokens
        tokens = list(token_counts.keys())
        sorted_tokens = sorted(tokens, key=lambda x: token_counts[x])[::-1]
        if max_size > 0:
            sorted_tokens = sorted_tokens[:max_size]
        # Add the remaining tokens to the dictionary
        for token in sorted_tokens:
            vocab.add_symbol(token)

        return vocab


def _make_tagged_tokens(sents, pad_idx):
    """Pad sentences to the max length and create the relevant tag"""
    lengths = [len(sent) for sent in sents]
    max_len = max(lengths)
    bsz = len(lengths)
    # Tensor containing the (right) padded tokens
    tokens = th.full((max_len, bsz), pad_idx).long()
    for i in range(bsz):
        tokens[:lengths[i], i] = th.LongTensor(sents[i])
    # Mask such that tag[i, b] = 1 iff lengths[b] < i
    lengths = th.LongTensor(lengths).view(1, -1)
    tag = th.gt(th.arange(max_len).view(-1, 1), lengths)
    # print (lengths, th.arange(max_len).view(-1, 1), tag)
    return tokens, tag


class MTDataset(data.Dataset):

    def __init__(self, vocab, prefix, src_lang="en", tgt_lang="fr"):
        # Attributes
        self.vocab = vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # Load from files
        src_file = prefix + "." + src_lang
        tgt_file = prefix + "." + tgt_lang
        self.src_txt = loadtxt(src_file)
        self.tgt_txt = loadtxt(tgt_file)
        # Check length
        self.length = len(self.src_txt)
        if self.length != len(self.tgt_txt):
            raise ValueError("Mismatched source and target length")
        # Append start/end of sentence token to the target
        for idx, tgt_sent in enumerate(self.tgt_txt):
            self.tgt_txt[idx] = f"<sos> {tgt_sent} <eos>"
        # Convert to indices
        self.src_idxs = [
            [self.vocab[tok] for tok in sent.split()] + [self.vocab["<eos>"]]
            for sent in self.src_txt
        ]
        self.tgt_idxs = [
            [self.vocab[tok] for tok in sent.split()]
            for sent in self.tgt_txt
        ]

    def __getitem__(self, i):
        return self.src_idxs[i], self.tgt_idxs[i]

    def __len__(self):
        return self.length


class MTDataLoader(data.DataLoader):
    """Special Dataloader for MT datasets

    Batches by number of sentences and/or tokens
    """

    def __init__(self, dataset, vocab, dynamic_tag=False, max_bsz=1, max_tokens=1000, shuffle=False):
    
        self.dataset = dataset
        self.max_bsz = max_bsz
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.vocab = vocab
        self.dynamic_tag = dynamic_tag
        if self.dynamic_tag:
            print("Training with Dynamic Mask.")
        # Order of batches

    def init_epoch(self):
        """Make batches that contain no more than
        `max_tokens` tokens and `max_bsz` samples"""
        N = len(self.dataset)
        if self.shuffle:
            self.order = th.randperm(N).numpy()
        else:
            self.order = th.arange(N).long().numpy()
        self.batches = []
        batch_size = max_src_tokens = max_tgt_tokens = 0
        current_batch = []
        pointer = 0
        while pointer < N:
            idx = self.order[pointer]
            src, tgt = self.dataset[idx]
            # Check whether adding this sample would bring us over
            # the size limit
            batch_size += 1
            max_src_tokens = max(max_src_tokens, len(src))
            max_tgt_tokens = max(max_tgt_tokens, len(tgt))
            tot_tokens = (max_src_tokens + max_tgt_tokens) * batch_size
            # If this is the case, wrap up current batch
            if batch_size > self.max_bsz or tot_tokens > self.max_tokens:
                if len(current_batch) > 0:
                    self.batches.append(current_batch)
                else:
                    # If this happens then there is one sample that is too big,
                    # just ignore it wth a warning
                    print(f"WARNING: ignoring sample {idx}"
                          "(too big for specified batch size)")
                    pointer += 1
                batch_size = max_src_tokens = max_tgt_tokens = 0
                current_batch = []
            else:
                current_batch.append(idx)
                pointer += 1
        # Add the last batch
        if len(current_batch) > 0:
            self.batches.append(current_batch)

                 

    def process_tokens(self, tag_dict):

        processed_tag_dict = {
            self.vocab[k] : v for k, v in tag_dict.items() if k in self.vocab.idxs
        }        

        return processed_tag_dict


    def __iter__(self):
        self.init_epoch()
        self.pos = 0
        return self

    def __len__(self):
        return len(self.batches)

    def get_batch(self, pos):
        samples = [self.dataset[i] for i in self.batches[pos]]
        src_sents = [src for src, _ in samples]
        if self.dynamic_tag:
            tgt_sents = [self.get_gtagged(tgt) for _, tgt in samples]
            # for tgt in tgt_sents:
            #     cnt = 0
            #     for k in tgt:
            #         print(k)
            #         cnt+=(k == 4)
            #     print ("count",cnt)
            tgt_sents = np.array(tgt_sents)
            selection = np.ones(len(tgt_sents), dtype=bool)
            selection[1:] = tgt_sents[1:] != tgt_sents[:-1]
            tgt_sents = tgt_sents[selection]
        else:
            tgt_sents = [tgt for _, tgt in samples]
        # Input tensor
        pad_idx = self.dataset.vocab["<pad>"]
        src_tokens, src_tag = _make_tagged_tokens(src_sents, pad_idx)
        tgt_tokens, tgt_tag = _make_tagged_tokens(tgt_sents, pad_idx)
        # print(sum(tgt_tokens==4))
        return src_tokens, src_tag, tgt_tokens, tgt_tag

    
    def get_gtagged(self, tgt_sent):
        
        output = []
        for tok in tgt_sent:
            if tok in self.p9_tags:
                output.append(self.get_random(tok, self.p9_tags[tok]))
            else:
                output.append(tok)

        return output
        
    def get_random(self, tok, prob):

        if np.random.uniform() < prob:
            return self.vocab["[GMASK]"]
        else:
            return tok

    def __next__(self):
        if self.pos >= len(self.batches):
            raise StopIteration()
        batch = self.get_batch(self.pos)
        self.pos += 1
        return batch




class MTNoisyDataset(data.Dataset):

    def __init__(self, vocab, prefix, src_lang="en", tgt_lang="fr"):
        # Attributes
        self.vocab = vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Load from files
        src_file = prefix + "." + src_lang
        tgt_file = prefix + "." + tgt_lang
        self.src_txt = loadtxt(src_file)
        self.tgt_txt = loadtxt(tgt_file)
        # Check length
        self.length = len(self.src_txt)
        if self.length != len(self.tgt_txt):
            raise ValueError("Mismatched source and target length")
        # Append start/end of sentence token to the target
        for idx, tgt_sent in enumerate(self.tgt_txt):
            self.tgt_txt[idx] = f"<sos> {tgt_sent} <eos>"
        # Convert to indices
        self.src_idxs = [
            [self.vocab[tok] for tok in sent.split()] + [self.vocab["<eos>"]]
            for sent in self.src_txt
        ]
        self.tgt_idxs = [
            [self.vocab[tok] for tok in sent.split()]
            for sent in self.tgt_txt
        ]

    def __getitem__(self, i):
        return self.src_idxs[i], self.tgt_idxs[i]

    def __len__(self):
        return self.length
