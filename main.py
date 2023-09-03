import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from models.models import *

if (not torch.cuda.is_available()):
    print("gpu not available")
    exit()

device = torch.device("cuda:0")

# Load data
lines = open("./data/tinyshakespeare/input.txt", "r").read()
vocab = sorted(list(set(lines)))
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

MASTER_CONFIG = {
    "vocab_size": len(vocab),
    "batch_size": 32,
    "context_window": 16,
    "d_model": 128,
    "epochs": 1000,
    "log_interval": 10,
}


# simple tokenization by characters
def encode(s):
    return [stoi[ch] for ch in s]


def decode(l):
    return ''.join([itos[i] for i in l])


def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    batch_data = train
    if split == 'val':
        batch_data = val

    if split == 'test':
        batch_data = test

    # pick random starting points
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i + context_window] for i in ix]).long()
    y = torch.stack([batch_data[i + 1:i + context_window + 1] for i in ix]).long()

    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()  # don't compute gradients for this function
def evaluate_loss(model, config=MASTER_CONFIG):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    losses = []
    start_time = time.time()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()

        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])
        logits, loss = model(xs, targets=ys)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model)
            losses += [x]
            if print_logs:
                print(
                    f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch) / config['log_interval'] :.3f}")
            start_time = time.time()

            if scheduler:
                print("lr: ", scheduler.get_lr())

    print("validation loss: ", losses[-1]['val'])
    return pd.DataFrame(losses)


def generate(model, config=MASTER_CONFIG, max_new_tokens=30):
    idx = torch.zeros(5, 1).long()
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        # call the model
        logits = model(idx[:, -config['context_window']:])

        last_time_step_logits = logits[
                                :, -1, :
                                ]  # all the batches (1), last time step, all the logits
        p = F.softmax(last_time_step_logits, dim=-1)  # softmax to get probabilities
        idx_next = torch.multinomial(
            p, num_samples=1
        )  # sample from the distribution to get the next token
        idx = torch.cat([idx, idx_next], dim=-1)  # append to the sequence
    return [decode(x) for x in idx.tolist()]


# load dataset
dataset = torch.tensor(encode(lines), dtype=torch.int8)
dataset = dataset.to(device)
torch.Size([1115394])

# load model
model = SimpleModel(MASTER_CONFIG)
model = model.to(device)

xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = model(xs, ys)

optimizer = torch.optim.Adam(model.parameters(), )

train_loss = train(model, optimizer)
inference_data = generate(model)
print(inference_data)

plt.plot(train_loss)
plt.show()
