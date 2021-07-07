# NeuReduce_pytorch
https://aclanthology.org/2020.findings-emnlp.56/


## Data pre-processing

```python
import torchtext.legacy as torchtext
from torchtext.legacy.data import Field, BucketIterator
```

```python
SRC = Field(tokenize=_tokenize,init_token='<sos>',eos_token='<eos>',pad_token='<pad>',lower=True,batch_first=True)
TRG = Field(tokenize=_tokenize,init_token='<sos>',eos_token='<eos>',pad_token='<pad>',lower=True,batch_first=True)
```
