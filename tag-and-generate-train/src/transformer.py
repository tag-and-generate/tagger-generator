from math import pi, sqrt
import torch as th
from torch import nn
from noisy import noisy

# Negative infinity constant
NEG_INF = -100000


def GlorotLinear(input_dim, output_dim):
    """Returns a Glorot initialized linear layer for optimal gradient flow"""
    linear = nn.Linear(input_dim, output_dim)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.constant_(linear.bias, 0)
    return linear


class MultiHeadAttention(nn.Module):
    """Multi head attention"""

    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        # Hyper-parameters
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be a multiple of n_heads")
        # Input projection layers
        self.query = GlorotLinear(self.embed_dim, self.embed_dim)
        self.key = GlorotLinear(self.embed_dim, self.embed_dim)
        self.value = GlorotLinear(self.embed_dim, self.embed_dim)
        # Output projection layers
        self.output = GlorotLinear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        queries,
        keys,
        values,
        in_mask=None,
        causal_masking=False,
        return_weights=False,
    ):
        """
        :param queries: Tensor of shape m x b x embed_dim where m is the length
            dimension and b the batch dimension
        :param keys: Tensor of shape n x b x embed_dim where n is the length
            dimension and b the batch dimension
        :param values: Tensor of shape n x b x embed_dim where n is the length
            dimension and b the batch dimension
        :param in_mask: n x b mask with 1 at positions that shouldn't be
            attended to (typically padding tokens)
        :param causal_masking: For each query position i, set the attention to
            all key positions j >i to 0, thus preventing the model from
            attending "to the future" (typically in unidirectional
            language models)
        :param return_weights: Return attention weights
        """
        m, bsz, _ = queries.size()
        n, _, _ = keys.size()
        # Project keys, queries and values (all of shape m/n x b x embed_dim)
        # Reshape the last dim as n_heads x head_dims
        q = self.query(queries).view(m, bsz, self.n_heads, self.head_dim)
        k = self.key(keys).view(n, bsz, self.n_heads, self.head_dim)
        v = self.value(values).view(n, bsz, self.n_heads, self.head_dim)
        # Compute attention potentials
        potentials = th.einsum("mbhd,nbhd->mnbh", [q, k])
        # Rescale by inverse sqrt of the dimension for well behaved softmax
        potentials /= sqrt(self.embed_dim)
        # Mask certain input positions
        if in_mask is not None:
            in_mask = in_mask.view(1, n, bsz, 1)
            potentials = potentials.masked_fill(in_mask, NEG_INF)
        # Causal masking: make it impossible to "attend to the future"
        if causal_masking:
            # We want causal_mask[i, j] = 1 if j > i
            causal_mask = th.triu(th.ones(m, n), diagonal=1).view(m, n, 1, 1)
            causal_mask = causal_mask.eq(1).to(potentials.device)
            potentials = potentials.masked_fill(causal_mask, NEG_INF)
        # Softmax over the input length n, differently for each head
        weights = nn.functional.softmax(potentials, dim=1)
        # Compute the pooled values
        pooled_v = th.einsum("mnbh,nbhd->mbhd", [weights, v]).contiguous()
        # Output projection
        output = self.output(pooled_v.view(m, bsz, -1))
        if return_weights:
            return output, weights
        else:
            return output


class FeedForwardTransducer(nn.Module):
    """Applies a 2-layer MLP to each position in a sequence"""

    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super(FeedForwardTransducer, self).__init__()
        # Hyper parameters
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # Layers
        self.layers = nn.Sequential(
            GlorotLinear(self.embed_dim, self.hidden_dim),  # Input projection
            nn.ReLU(),                                      # Activation
            nn.Dropout(p=self.dropout),                     # Dropout
            GlorotLinear(self.hidden_dim, self.embed_dim),  # Output projection
        )

    def forward(self, x):
        """
        :param x: Tensor of shape n x b x embed_dim where n is the length
            dimension and b the batch dimension
        """
        return self.layers(x)


class EncoderLayer(nn.Module):
    """Transformer encoder layer"""

    def __init__(self, embed_dim, n_heads, hidden_dim, dropout=0.0):
        super(EncoderLayer, self).__init__()
        # Hyper parameters
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # Sub-layers
        # Self attention
        self.layer_norm_self_att = nn.LayerNorm(embed_dim)
        self.self_att = MultiHeadAttention(embed_dim, n_heads)
        self.drop_self_att = nn.Dropout(p=dropout)
        # Feed forward
        self.layer_norm_ff = nn.LayerNorm(embed_dim)
        self.ff = FeedForwardTransducer(embed_dim, hidden_dim, dropout)
        self.drop_ff = nn.Dropout(p=dropout)

    def forward(self, x, src_mask=None):
        """
        :param x: Tensor of shape n x b x embed_dim where n is the length
            dimension and b the batch dimension
        :param src_mask: Mask of shape n x b indicating padding tokens in
            the source sentences (for masking in self-attention)
        """
        # Self attention
        x_normed = self.layer_norm_self_att(x)
        h_self_att = self.self_att(
            queries=x_normed,
            keys=x_normed,
            values=x_normed,
            in_mask=src_mask,
        )
        x = x + self.drop_self_att(h_self_att)
        # Feed-forward transform
        x_normed = self.layer_norm_ff(x)
        h_ff = self.ff(x_normed)
        return x + self.drop_ff(h_ff)


class DecoderLayer(nn.Module):
    """Transformer decoder layer"""

    def __init__(self, embed_dim, n_heads, hidden_dim, dropout=0.0):
        super(DecoderLayer, self).__init__()
        # Hyper parameters
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # Sub-layers
        # Self attention
        self.layer_norm_self_att = nn.LayerNorm(embed_dim)
        self.self_att = MultiHeadAttention(embed_dim, n_heads)
        self.drop_self_att = nn.Dropout(p=dropout)
        # Encoder attention
        self.layer_norm_enc_att = nn.LayerNorm(embed_dim)
        self.enc_att = MultiHeadAttention(embed_dim, n_heads)
        self.drop_enc_att = nn.Dropout(p=dropout)
        # Feed forward
        self.layer_norm_ff = nn.LayerNorm(embed_dim)
        self.ff = FeedForwardTransducer(embed_dim, hidden_dim, dropout)
        self.drop_ff = nn.Dropout(p=dropout)

    def forward(self, x, encodings, src_mask=None):
        """
        :param x: Tensor of shape n x b x embed_dim where n is the length
            dimension and b the batch dimension
        :param src_mask: Mask of shape n x b indicating padding tokens in
            the source sentences (for masking in encoder-attention)
        """
        # Self attention
        x_normed = self.layer_norm_self_att(x)
        h_self_att = self.self_att(
            queries=x_normed,
            keys=x_normed,
            values=x_normed,
            causal_masking=True,  # Don't attend to the future
        )
        x = x + self.drop_self_att(h_self_att)
        # Encoder attention
        x_normed = self.layer_norm_enc_att(x)
        h_enc_att = self.enc_att(
            queries=x_normed,
            keys=encodings,
            values=encodings,
            in_mask=src_mask,
        )
        x = x + self.drop_enc_att(h_enc_att)
        # Feed-forward transform
        x_normed = self.layer_norm_ff(x)
        h_ff = self.ff(x_normed)
        return x + self.drop_ff(h_ff)

    def decode_step(
        self,
        x,
        encodings,
        state,
        src_mask=None,
    ):
        """
        This performs a forward pass on a single vector.
        This is used during decoding.

        :param x: Tensor of shape 1 x b x embed_dim where b is the batch
            dimension. This is the input at the current position only
        :param src_mask: Mask of shape n x b indicating padding tokens in
            the source sentences (for masking in self-attention)
        :param state: This is either None or a n x b x embed_dim tensor
            containing the inputs to the self attention layers up until
            this position. This method returns an updated state
        """
        # Self attention
        x_normed = self.layer_norm_self_att(x)
        # Update state
        if state is None:
            state = x_normed
        else:
            state = th.cat([state, x_normed], dim=0)
        h_self_att = self.self_att(
            queries=x_normed,
            keys=state,
            values=state,
        )
        x = x + self.drop_self_att(h_self_att)
        # Encoder attention
        x_normed = self.layer_norm_enc_att(x)
        h_enc_att = self.enc_att(
            queries=x_normed,
            keys=encodings,
            values=encodings,
            in_mask=src_mask,
        )
        x = x + self.drop_enc_att(h_enc_att)
        # Feed-forward transform
        x_normed = self.layer_norm_ff(x)
        h_ff = self.ff(x_normed)
        return h_ff + self.drop_ff(x), state


def sin_embeddings(max_pos, dim):
    """Returns sinusoidal embedings(for position embeddings)"""
    # Scale for each dimension
    dim_scale = 2 * (th.arange(dim) / 2).long().float() / dim
    dim_scale = th.pow(th.full((dim,), 10000.0), dim_scale).view(1, -1)
    # Phase to change sine to cosine every other dim
    phase = th.zeros((1, dim))
    phase[0, 1::2] = pi / 2
    # Position value
    pos = th.arange(max_pos).float().view(-1, 1)
    # Embeddings
    embeds = th.sin(pos / dim_scale + phase)
    return embeds


class Transformer(nn.Module):
    """The full transformer model"""

    def __init__(
        self,
        n_layers,
        embed_dim,
        hidden_dim,
        n_heads,
        vocab,
        dropout=0.0,
        word_dropout=0.0,
    ):
        """
        :param n_layers: Number of layers (both encoder and decoder)
        :param embed_dim: Embedding dimension (dimension throughout the model)
        :param hidden_dim: Dimension of the hidden layer in position-wise
            feed-forward layers
        :param n_heads: Number of attention heads
        :param vocab: Vocabulary object (see data.py)
        :param dropout: Dopout probability
        """
        super(Transformer, self).__init__()
        # Hyper-parameters
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.vocab = vocab
        # Token embeddings (this will be shared for encoder/decoder)
        self.embeds = nn.Embedding(len(vocab), embed_dim, 0)
        nn.init.normal_(self.embeds.weight, std=1/sqrt(embed_dim))
        self.embed_drop = nn.Dropout(p=dropout)
        self.word_drop = nn.Dropout(p=word_dropout)
        # Positional embeddings
        self.pos_embeds = sin_embeddings(2048, embed_dim)
        # Encoder Layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, n_heads, hidden_dim, dropout=dropout)
            for l in range(n_layers)
        ])
        # Final encoder layer norm
        self.layer_norm_enc = nn.LayerNorm(embed_dim)
        # Output proj (this is important because the embeddings are tied)
        # and the output has been "layer-normalized".
        # this layer can adjust the scale before the logits.
        self.out_proj = GlorotLinear(embed_dim, embed_dim)
        # Decoder Layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, n_heads, hidden_dim, dropout=dropout)
            for l in range(n_layers)
        ])
        # Final decoder layer norm
        self.layer_norm_dec = nn.LayerNorm(embed_dim)
        # Output projection for the logits
        self.logits = GlorotLinear(embed_dim, len(vocab))
        # Share embedding and softmax weights
        self.logits.weight = self.embeds.weight

    def encode(self, src_tokens, src_mask=None):
        """
        This encodes a batch of tokens (for feeding into the decoder)

        :param src_tokens: Tensor of integers of shape n x b representing
            the source tokens
        :param src_mask: Tensor of shape n x b identifying the padding
            tokens for masking
        """
        # Embed and rescale
        x = self.embeds(src_tokens) * sqrt(self.embed_dim)
        
        # Apply dropout
        x = self.embed_drop(x)
        # Word dropout
        x = x * self.word_drop(th.ones_like(src_tokens).float()).unsqueeze(-1)
        # Add position embedding
        pos_offset = self.pos_embeds[:x.size(0)].view(-1, 1, self.embed_dim)
        x += pos_offset.to(x.device).detach()
        # Run through the encoder
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)
        # Layer normalize
        # (to prevent all the residual connections from blowing up)
        return self.layer_norm_enc(x)

    def forward(self, src_tokens, tgt_tokens, src_mask=None):
        """
        Returns a tensor log_p of shape m x b x |V| where log_p[i, k, w]
        corresponds to the log probability of word w being at position i
        in the bth target sentence (conditioned on the bth source sentence
        and all the tokens at positions <i).

        :param src_tokens: Tensor of integers of shape n x b representing
            the source tokens
        :param tgt_tokens: Tensor of integers of shape m x b representing
            the target tokens
        :param src_mask: Tensor of shape n x b identifying the padding
            tokens for masking
        """
        
        # Encode source tokens
        encodings = self.encode(src_tokens, src_mask)
        # Embed target tokens
        h = self.embeds(tgt_tokens) * sqrt(self.embed_dim)
        h = self.embed_drop(h)
        # Word dropout
        h = h * self.word_drop(th.ones_like(tgt_tokens).float()).unsqueeze(-1)
        # Add position embeddings
        pos_offset = self.pos_embeds[:h.size(0)].view(-1, 1, self.embed_dim)
        h += pos_offset.to(h.device).detach()
        # Pass through all decoder layers
        for layer in self.decoder_layers:
            h = layer(h, encodings, src_mask=src_mask)
        # Final layer norm so things don't blow up
        h = self.layer_norm_dec(h)
        # Output proj (into the embedding dimension).
        # This is necessary to make the model expressive enough since the
        # softmax weights are shared with the embeddings
        h = self.out_proj(h)
        # obtain logits for every word
        logits = self.logits(h)
        # Return log probs
        return nn.functional.log_softmax(logits, dim=-1)

    def decode_step(
        self,
        tgt_token,
        encodings,
        states,
        src_mask=None,
    ):
        """
        This performs a forward pass on a single vector.
        This is used during decoding.

        :param x: Tensor of shape 1 x b x embed_dim where b is the batch
            dimension. This is the input at the current position only
        :param src_mask: Mask of shape n x b indicating padding tokens in
            the source sentences (for masking in self-attention)
        :param states: This is a list of either None or a n x b x embed_dim
            tensors containing the inputs to each self attention layers up
            until this position. This method returns an updated state.
        """
        new_states = []
        h = self.embeds(tgt_token) * sqrt(self.embed_dim)
        h = self.embed_drop(h)
        # Word dropout
        h = h * self.word_drop(th.ones_like(tgt_token).float()).unsqueeze(-1)
        # Add position embedding
        pos = 0 if states[0] is None else states[0].size(0)
        pos_offset = self.pos_embeds[pos].view(1, 1, -1)
        h += pos_offset.to(h.device).detach()
        # Pass through all layers
        for layer, state in zip(self.decoder_layers, states):
            h, new_state = layer.decode_step(
                h,
                encodings,
                state,
                src_mask=src_mask,
            )
            new_states.append(new_state)
        # Final layer norm so things don't blow up
        h = self.layer_norm_dec(h)
        # Output proj
        h = self.out_proj(h)
        logits = self.logits(h)
        # Log prob at this position
        log_p = nn.functional.log_softmax(logits, dim=-1)
        return log_p, new_states

    def initial_state(self):
        """Returns the initial state for decoding (a list of None)"""
        return [None for _ in range(self.n_layers)]
