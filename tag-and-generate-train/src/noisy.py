import numpy as np
import torch
from data import _make_tagged_tokens
import itertools

# TODO: Add reference

def word_shuffle(vocab, x, k):   # slight shuffle such that |sigma[i]-i| <= k
    base = torch.arange(x.size(0), dtype=torch.float).repeat(x.size(1), 1).t()
    inc = (k+1) * torch.rand(x.size())
    inc[x == vocab['<sos>']] = 0     # do not shuffle the start sentence symbol
    inc[x == vocab['<pad>']] = k+1  # do not shuffle end paddings
    inc[x == vocab['<eos>']] = k+1
    _, sigma = (base + inc).sort(dim=0)
    return x[sigma, torch.arange(x.size(1))]

def word_drop(vocab, x, p):     # drop words with probability p
    x_ = []
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = True  # do not drop the start sentence symbol
        sent = [w for j, w in enumerate(words) if keep[j]]
        sent += [vocab['<pad>']] * (len(words)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def word_blank(vocab, x, p):     # blank words with probability p
    blank = (torch.rand(x.size(), device=x.device) < p) & \
        (x != vocab['<pad>']) & (x != vocab['<sos>']) & (x != vocab['<eos>'])
    x_ = x.clone()
    x_[blank] = vocab['<unk>']
    return x_

def word_substitute(vocab, x, p):     # substitute words with probability p
    keep = (torch.rand(x.size(), device=x.device) > p) | \
        (x == vocab['<sos>']) | (x == vocab['<pad>']) | (x == vocab['<eos>']) | (x == vocab['[GMASK]'])
    x_ = x.clone()
    x_.random_(0, len(vocab))
    x_[keep] = x[keep]
    return x_

def add_gtag(vocab, x, p):     # drop words with probability p
    x_ = []
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        add = np.random.rand(len(words)) < p
        add[-1] = False
        # sent = [[w , vocab['▁['], vocab['GMASK'] , vocab[']']] if add[j] else [w] for j, w in enumerate(words)]
        sent = [[w , vocab[f'[GMASK{j//3}]']] if add[j] else [w] for j, w in enumerate(words)]
        sent = list(itertools.chain.from_iterable(sent)) + [vocab['<pad>']]
        x_.append(sent)
    sent, _ = _make_tagged_tokens(x_, vocab['<pad>'])
    return sent.to(x.device)

def add_intelligent_gtag(vocab, x, p):     # drop words with probability p
    x_ = []
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        add = np.random.rand(len(words)) < p
        add[-1] = False
        sent = [[w, vocab['GMASK']] if add[j] and w==vocab['GMASK'] else [w] for j, w in enumerate(words)]
        sent = list(itertools.chain.from_iterable(sent)) + [vocab['<pad>']]
        x_.append(sent)
    sent, _ = _make_tagged_tokens(x_, vocab['<pad>'])
    return sent.to(x.device)


def intelligent_word_shuffle(vocab, x, k):   # slight shuffle such that |sigma[i]-i| <= k
    base = torch.arange(x.size(0), dtype=torch.float).repeat(x.size(1), 1).t()
    inc = (k+1) * torch.rand(x.size())
    for j in range(x.size(1)):
        for i in range(x.size(0)):
            do_shuf = 0
            for l in range(k//2):
                if x[max(i-l,0)][j] == vocab['GMASK']:
                    do_shuf = 1
            for l in range(k//2):
                if x[min(i+l,x.size(0)-1)][j] == vocab['GMASK']: 
                    do_shuf = 1
            inc[i][j] *= do_shuf
    inc[x == vocab['<sos>']] = 0     # do not shuffle the start sentence symbol
    inc[x == vocab['<pad>']] = k+1  # do not shuffle end paddings
    inc[x == vocab['<eos>']] = k+1
    _, sigma = (base + inc).sort(dim=0)
    return x[sigma, torch.arange(x.size(1))]


def noisy(vocab, x, drop_prob, blank_prob, sub_prob, shuffle_dist, add_gtag_prob, add_int_gtag_prob):
    if drop_prob > 0:
        x = word_drop(vocab, x, drop_prob)
    if blank_prob > 0:
        x = word_blank(vocab, x, blank_prob)
    if sub_prob > 0:
        x = word_substitute(vocab, x, sub_prob)
    if add_int_gtag_prob > 0:
        x = add_intelligent_gtag(vocab, x, add_gtag_prob)
        x = intelligent_word_shuffle(vocab, x, 3)
    if add_gtag_prob > 0:
        x = add_gtag(vocab, x, add_gtag_prob)
    if shuffle_dist > 0:
        x = word_shuffle(vocab, x, shuffle_dist)
    return x
