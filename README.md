# NeuReduce_pytorch
https://aclanthology.org/2020.findings-emnlp.56/

### Find out markdown
1. [How to Data-processing](https://github.com/DolceLatte/NeuReduce_pytorch/blob/main/How%20to%20Data%20pre-processing.md)<br/>
2. [Implementation of Transformer](https://github.com/DolceLatte/NeuReduce_pytorch/blob/main/Model_summary.md)<br/>
3. [Implementation of LSTM Based on Attention](https://github.com/DolceLatte/NeuReduce_pytorch/blob/main/LSTM.md)<br/>

### Comparative evaluation result
|Method|Correct|Ratio|Result Length|Solving Time
|---|---|---|---|---|
|LSTM-Attention|7530|69.20%|6.8|**0.007s**|
|Transformer|5040|**50.40%**|7.3|0.026s|

#### Example Expression
MBA expression : 2*(x&(y|z))+(~x|~z)-2*(y^z)+(~y^z)-2*(y^~z)-(x&y) <br/>
trg : -(~x&(y^z))<br/>
#### Results
Transformer predicted trg : -(~x&(y^z))<br/>
LSTM-Attention predicted trg : -(~x&(y^z))<br/>

#### Qsynth Expression
MBA expression : <br/>
trg : <br/>
#### Results
Transformer predicted trg :<br/>
LSTM-Attention predicted trg : <br/>



