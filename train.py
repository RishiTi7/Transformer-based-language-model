# chuncks of code

with open('input.txt','r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters", len(text))

print(text[:1000])


# here are unique chars in the text above

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


# create a mapping from characters to integers(string tokenizer) opeAi uses tiktoken

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l:''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))



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

print('-----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target is: {target}")