import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        B = q.size(0)
        q = self.w_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~mask, -1e9)
        out = F.softmax(scores, dim=-1) @ v

        out = out.transpose(1, 2).reshape(B, -1, self.h * self.d_k)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop(self.attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, h=8, d_ff=2048, n_layers=6,
                 dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)])

        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        self.generator.weight = self.embed.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)

    def embedding(self, x):
        return self.drop(self.pos(self.embed(x) * math.sqrt(self.d_model)))

    def make_masks(self, src, tgt):
        src_mask = (src != self.pad_idx)[:, None, None, :]
        T = tgt.size(1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device))
        tgt_mask = (tgt != self.pad_idx)[:, None, None, :] & causal
        return src_mask, tgt_mask

    def encode(self, src, src_mask):
        x = self.embedding(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.embedding(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_masks(src, tgt)
        memory = self.encode(src, src_mask)
        return self.generator(self.decode(tgt, memory, src_mask, tgt_mask))


@torch.no_grad()
def beam_search(model, src, bos_idx, eos_idx, beam_size=4, alpha=0.6, extra_len=50):

    model.eval()
    src_mask = (src != model.pad_idx)[:, None, None, :]
    memory = model.encode(src, src_mask)

    seqs = torch.full((1, 1), bos_idx, dtype=torch.long, device=src.device)
    scores = torch.zeros(1, device=src.device)
    finished = []

    def lp(length):
        return ((5 + length) / 6) ** alpha

    for _ in range(src.size(1) + extra_len):
        k = seqs.size(0)
        tgt_mask = torch.tril(torch.ones(seqs.size(1), seqs.size(1), dtype=torch.bool,
                                         device=src.device))
        out = model.decode(seqs, memory.expand(k, -1, -1), src_mask, tgt_mask)
        logp = F.log_softmax(model.generator(out[:, -1]), dim=-1)

        cand = (scores.unsqueeze(1) + logp).view(-1)
        top, idx = cand.topk(2 * beam_size)

        new_seqs, new_scores = [], []
        for s, i in zip(top.tolist(), idx.tolist()):
            b, tok = divmod(i, logp.size(1))
            if tok == eos_idx:
                finished.append((s / lp(seqs.size(1)), seqs[b, 1:].tolist()))
            else:
                new_seqs.append(torch.cat([seqs[b], torch.tensor([tok], device=src.device)]))
                new_scores.append(s)
            if len(new_seqs) == beam_size:
                break

        if len(finished) >= beam_size or not new_seqs:
            break
        seqs = torch.stack(new_seqs)
        scores = torch.tensor(new_scores, device=src.device)

    if not finished:
        finished = [(s / lp(seqs.size(1)), q[1:].tolist()) for s, q in zip(scores.tolist(), seqs)]
    return max(finished, key=lambda f: f[0])[1]
