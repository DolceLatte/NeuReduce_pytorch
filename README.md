# NeuReduce_pytorch
https://aclanthology.org/2020.findings-emnlp.56/

### Find out markdown
1. [How to Data-processing](https://github.com/DolceLatte/NeuReduce_pytorch/blob/main/How%20to%20Data%20pre-processing.md)<br/>
2. [Implementation of Transformer](https://github.com/DolceLatte/NeuReduce_pytorch/blob/main/Model_summary.md)<br/>
3. [Implementation of LSTM Based on Attention](https://github.com/DolceLatte/NeuReduce_pytorch/blob/main/LSTM.md)<br/>

### Comparative evaluation result
|Method|Correct|Ratio|Result Length|Solving Time
|---|---|---|---|---|
|Transformer(Paper)|7824|78.24%|18.02|0.43s|
|Transformer(Ours)|7102|71.00%|16.2|0.055s|

#### Example Expression
MBA expression : 2*(x&(y|z))+(~x|~z)-2*(y^z)+(~y^z)-2*(y^~z)-(x&y) <br/>
trg : -(~x&(y^z))<br/>
#### Results
Transformer predicted trg : -(~x&(y^z))<br/>
LSTM-Attention predicted trg : -(~x&(y^z))<br/>
