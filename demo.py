import math
import torch
import torch.nn.functional as F
from model import MultiHeadAttention, PositionalEncoding, Transformer
from train import noam

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)


def title(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


title("1. Positional encoding (d_model=8, ilk 4 pozisyon)")
pe = PositionalEncoding(8).pe[:4]
print(pe)
print("cift sutunlar sin, tek sutunlar cos; sutun index'i arttikca frekans dusuyor")
print("pozisyon 0: sin(0)=0, cos(0)=1 ->", pe[0].tolist())

title("2. Maskeler")
model = Transformer(vocab_size=20, d_model=16, h=2, d_ff=32, n_layers=1, dropout=0.0)
src = torch.tensor([[5, 6, 7, 0, 0]])
tgt = torch.tensor([[1, 8, 9, 10]])
src_mask, tgt_mask = model.make_masks(src, tgt)
print("src:", src.tolist(), "-> src_mask", tuple(src_mask.shape))
print(src_mask[0, 0].int())
print("\ntgt:", tgt.tolist(), "-> tgt_mask", tuple(tgt_mask.shape), "(satir = query, sutun = key)")
print(tgt_mask[0, 0].int())

title("3. Scaled dot-product attention'i elle hesaplayip modulle karsilastirma")
mha = MultiHeadAttention(d_model=8, h=2)
x = torch.randn(1, 4, 8)
mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]

q = mha.w_q(x).view(1, 4, 2, 4).transpose(1, 2)
k = mha.w_k(x).view(1, 4, 2, 4).transpose(1, 2)
v = mha.w_v(x).view(1, 4, 2, 4).transpose(1, 2)
print("q:", tuple(q.shape), "-> (batch, head, T, d_k)")
scores = q @ k.transpose(-2, -1) / math.sqrt(4)
weights = F.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
print("\nhead 0 attention agirliklari (causal mask'li):")
print(weights[0, 0])
print("satir toplamlari:", weights[0, 0].sum(-1).tolist())
manual = mha.w_o((weights @ v).transpose(1, 2).reshape(1, 4, 8))
print("elle hesap == modul cikti:", torch.allclose(manual, mha(x, x, x, mask), atol=1e-6))

title("4. Neden sqrt(d_k)? (d_k = 64)")
d_k = 64
q = torch.randn(10000, d_k)
k = torch.randn(10000, d_k)
raw = (q * k).sum(-1)
print(f"dot product std, olceksiz : {raw.std():.2f}  (~sqrt(d_k) = {math.sqrt(d_k):.0f})")
print(f"dot product std, /sqrt(d_k): {(raw / math.sqrt(d_k)).std():.2f}")
logits = torch.randn(8, d_k) @ torch.randn(d_k)
print("\nolceksiz softmax :", F.softmax(logits, -1))
print("olcekli softmax  :", F.softmax(logits / math.sqrt(d_k), -1))
print("(olceksizde dagilim tek elemana yigiliyor, gradyan kayboluyor)")

title("5. Decoder gelecegi gormuyor mu?")
model.eval()
src_a = torch.tensor([[5, 6, 7]])
tgt_a = torch.tensor([[1, 8, 9, 10]])
tgt_b = torch.tensor([[1, 8, 9, 15]])
with torch.no_grad():
    out_a = model(src_a, tgt_a)
    out_b = model(src_a, tgt_b)
same = [torch.allclose(out_a[0, t], out_b[0, t], atol=1e-6) for t in range(4)]
print("tgt'nin sadece SON token'i degistirildi")
print("pozisyon bazinda cikti ayni mi:", same)
print("(ilk 3 pozisyon ayni, sadece son degisir)")

title("6. Padding sonucu etkiliyor mu?")
with torch.no_grad():
    out_1 = model(torch.tensor([[5, 6, 7]]), tgt_a)
    out_2 = model(torch.tensor([[5, 6, 7, 0, 0]]), tgt_a)
print("padding'li ve padding'siz kaynak ayni cikti veriyor:", torch.allclose(out_1, out_2, atol=1e-5))

title("7. Tam model: sekiller ve parametreler (paper base konfigurasyonu)")
base = Transformer(vocab_size=37000)
src = torch.randint(3, 37000, (2, 7))
tgt = torch.randint(3, 37000, (2, 5))
with torch.no_grad():
    logits = base.eval()(src, tgt)
print("src", tuple(src.shape), "tgt", tuple(tgt.shape), "-> logits", tuple(logits.shape))
print(f"parametre sayisi: {sum(p.numel() for p in base.parameters()):,}")
print("embedding ve cikis katmani ayni tensor mu:",
      base.generator.weight.data_ptr() == base.embed.weight.data_ptr())

title("8. Learning rate schedule (d_model=512, warmup=4000)")
for step in (1, 1000, 2000, 4000, 8000, 20000, 100000):
    print(f"step {step:>6}: lr = {noam(step, 512, 4000):.6f}")
print("lr 4000. step'e kadar lineer artiyor, sonra 1/sqrt(step) ile dusuyor")
