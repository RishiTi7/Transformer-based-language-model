# chuncks of code

with open('input.txt','r', encoding='utf-8') as f:
    text = f.read()

# print("length of dataset in characters", len(text))

# print(text[:1000])


# here are unique chars in the text above

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
# print(vocab_size)


# create a mapping from characters to integers(string tokenizer) opeAi uses tiktoken

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode("hii there"))
# print(decode(encode("hii there")))



# encode the entire text dataset & maybe 0 is a new line char and 1 is a space
import torch 
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])



# spilitting the data set into train and validation sets

n = int(0.9*len(data)) # 90%
train_data = data[:n]
val_data = data[n:]

# we always train the tranformer in chunks (blocks of data)

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is: {target}")


# Batch dimension    

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb,yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target is: {target}")


import torch 
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self , vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C) # arranging acc to pytorch B C T
            targets = targets.view(B*T)   
            loss = F.cross_entropy(logits, targets) # how well can we predict the next char based on loss(targets)

        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        # evaluating all the batch dimensions in the time dimension BT +1
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[:,-1,:] # plucking out the last time dimension
            probs = F.softmax(logits,dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)    

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))#gives out random sequence


# create a pytorch optimizer

optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

batch_size = 32
for steps in range(10000):
    xb,yb = get_batch('train')
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    # print(loss.item())


print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))


# mathematical trick in self attention
# efficient implementation
# the tokens should talk in 3 2 1 = 4 not 5 6 7 = 4 coz we are basing this on history not on prediction of history
torch.manual_seed(1337)
B,T,C = 4,8,32
x = torch.randn(B,T,C)
x.shape

#bag of words
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev,0)


# @ batch matrix multiplier

wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim = True)
xbow2 = wei @ x
torch.allclose(xbow,xbow2)

# way 2

tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril ==0,float('-inf')) # this will tell not to talk to future
wei = F.softmax(wei, dim =1)
xbow3 = wei @ x
torch.allclose(xbow,xbow2)

#tril = triangle
torch.tril(torch.ones(3,3))
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
a = a / torch.sum(a, 1, keepdim = True)
b = torch.randint(0,10,(3,2)).float()
c = a@b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('c=')
print(c)

