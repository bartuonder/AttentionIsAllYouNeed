import argparse
import os
import random
import urllib.request
import torch
import torch.nn.functional as F
from model import Transformer, beam_search

PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join("data", "tinyshakespeare.txt")


def load_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"downloading {URL}")
        try:
            urllib.request.urlretrieve(URL, DATA_PATH)
        except urllib.error.URLError as e:
            raise SystemExit(f"download failed ({e.reason}); fetch it manually:\n"
                             f"  curl -L {URL} -o {DATA_PATH}")
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i + N_SPECIAL for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text])
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], chars


def decode(ids, chars):
    return "".join(chars[i - N_SPECIAL] for i in ids if i >= N_SPECIAL)


def get_batch(data, batch_size, block_size, device):
    # seq2seq on a text stream: source = a block of text, target = the block that follows it
    starts = [random.randint(0, len(data) - 2 * block_size) for _ in range(batch_size)]
    src = torch.stack([data[i: i + block_size] for i in starts])
    tgt = torch.stack([data[i + block_size: i + 2 * block_size] for i in starts])
    bos = torch.full((batch_size, 1), BOS)
    eos = torch.full((batch_size, 1), EOS)
    tgt_in = torch.cat([bos, tgt], dim=1)
    tgt_out = torch.cat([tgt, eos], dim=1)
    return src.to(device), tgt_in.to(device), tgt_out.to(device)


def noam(step, d_model, warmup):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


@torch.no_grad()
def evaluate(model, data, args, device, n_batches=20):
    model.eval()
    loss = acc = 0.0
    for _ in range(n_batches):
        src, tgt_in, tgt_out = get_batch(data, args.batch_size, args.block_size, device)
        logits = model(src, tgt_in)
        loss += F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.view(-1),
                                ignore_index=PAD).item()
        acc += (logits.argmax(-1) == tgt_out).float().mean().item()
    return loss / n_batches, acc / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="paper's base config (d_model=512, N=6)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=64, help="chars in source and in target")
    ap.add_argument("--eval_every", type=int, default=500)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    train_data, val_data, chars = load_data()
    vocab_size = len(chars) + N_SPECIAL
    print(f"chars={len(chars)} vocab={vocab_size} train={len(train_data):,} val={len(val_data):,}")

    if args.base:
        cfg = dict(d_model=512, h=8, d_ff=2048, n_layers=6, dropout=0.1)
        warmup = 4000
    else:
        cfg = dict(d_model=128, h=4, d_ff=512, n_layers=2, dropout=0.1)
        warmup = 400

    model = Transformer(vocab_size, pad_idx=PAD, **cfg).to(device)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: noam(max(s, 1), cfg["d_model"], warmup))

    for step in range(1, args.steps + 1):
        model.train()
        src, tgt_in, tgt_out = get_batch(train_data, args.batch_size, args.block_size, device)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.view(-1),
                               ignore_index=PAD, label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.6f}")
        if step % args.eval_every == 0 or step == args.steps:
            val_loss, val_acc = evaluate(model, val_data, args, device)
            print(f"step {step:5d}  val_loss {val_loss:.4f}  val_char_acc {val_acc:.3f}")

    model.eval()
    for _ in range(3):
        src, _, tgt_out = get_batch(val_data, 1, args.block_size, device)
        pred = beam_search(model, src, BOS, EOS)
        print("src ", repr(decode(src[0].tolist(), chars)))
        print("pred", repr(decode(pred, chars)))
        print("gold", repr(decode(tgt_out[0].tolist(), chars)))


if __name__ == "__main__":
    main()
