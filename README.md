# NeuReduce_pytorch
https://aclanthology.org/2020.findings-emnlp.56/


## Data pre-processing

###### Input Data Sample : -2*(~x&y)-(~y)+2*(~x)-y+2 
###### Output Data Sample : -2*(x|y)+1 

```python
import torchtext.legacy as torchtext
from torchtext.legacy.data import Field, BucketIterator
```

```python
SRC = Field(tokenize=_tokenize,init_token='<sos>',eos_token='<eos>',pad_token='<pad>',lower=True,batch_first=True)
TRG = Field(tokenize=_tokenize,init_token='<sos>',eos_token='<eos>',pad_token='<pad>',lower=True,batch_first=True)
```

```python
exprs = torchtext.data.TabularDataset(
    path='./dataset.csv',
    format='csv',
    fields=[
        ('src', SRC),
        ('trg', TRG)
    ]
)

train_data, valid_data = exprs.split(split_ratio=0.8)

print(f'Total {len(exprs)} samples.')
print(f'Total {len(train_data)} train samples.')
print(f'Total {len(valid_data)} valid samples.')

print()
print(*exprs.examples[0].src, sep='')
print(*exprs.examples[0].trg, sep='')
```

```python
# Build vocab only from the training set, which can prevent information leakage
SRC.build_vocab(train_data)
TRG.build_vocab(train_data)
print(f'Total {len(SRC.vocab)} unique tokens in source vocabulary')
print(f'Total {len(TRG.vocab)} unique tokens in target vocabulary')
```

```python
batch_size = 128
device = torch.device('cuda')

train_iter, valid_iter = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=batch_size,
    sort=False,
    device=device
)
```

```python
print(SRC.vocab.stoi)
```
