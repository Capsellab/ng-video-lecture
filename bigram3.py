import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000 # Inserting a single self-attention block to our network: 4
eval_interval = 300
learning_rate = 1e-3 # Inserting a single self-attention block to our network: 4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # New things Number 1: GPU
eval_iters = 200
n_embd = 32 # Minor code cleanup: 2
# ------------

torch.manual_seed(1337)

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
## wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# import urllib.request
#
# url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# save_name='input.txt'
#
# urllib.request.urlretrieve(url, save_name)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # New things Number 1: GPU
    return x, y

@torch.no_grad() # this is just telling PyTorch that everything that happens inside this function, will not call that backward on.
def estimate_loss(): # New things Number 2: average loss function
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module): # Inserting a single self-attention block to our network: 1
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)                                 # (B,T,C)
        q = self.query(x)                               # (B,T,C)
        # compute attention scores ("affinities")
        wei =  q @ k.transpose(-2, -1) * C**-0.5        # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1)                    # (B,T,T)
        # perform the weighted aggregation of the values
        v = self.value(x)                               # (B,T,C)
        out = wei @ v                                   # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out

class MultiHeadAttention(nn.Module): # Multi-headed self-attention: 1
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self): # Minor code cleanup: 1
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Minor code cleanup: 2
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Positional encoding: 1
        #self.sa_head = Head(n_embd) # Inserting a single self-attention block to our network: 2
        self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self-attention # Multi-headed self-attention: 2
        self.lm_head = nn.Linear(n_embd, vocab_size) # Minor code cleanup: 3

    def forward(self, idx, targets=None):
        B, T = idx.shape # Positional encoding: 2

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) # Minor code cleanup: 3
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) # Positional encoding: 2
        x = tok_emb + pos_emb # (B,T,C) + (1,T,C) --> (B,T,C) # Positional encoding: 2
        #x = self.sa_head(x) # apply one head of self-attention. (B,T,C) # Inserting a single self-attention block to our network: 2
        x = self.sa_heads(x) # apply multi heads of self-attention. (B,T,C) # Multi-headed self-attention: 2
        logits = self.lm_head(x) # (B,T,vocab_size) # Positional encoding: 2

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # Inserting a single self-attention block to our network: 3
            # get the predictions (do forward function)
            logits, loss = self(idx_cond) # Inserting a single self-attention block to our network: 3
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

model = BigramLanguageModel() # Minor code cleanup: 1
m = model.to(device) # New things Number 1: GPU

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss() # New things Number 2: average loss function
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # New things Number 1: GPU
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
