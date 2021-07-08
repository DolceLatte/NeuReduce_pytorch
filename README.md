# NeuReduce_pytorch
https://aclanthology.org/2020.findings-emnlp.56/

### Find out markdown
1. [How to Data-processing](https://github.com/DolceLatte/NeuReduce_pytorch/blob/main/How%20to%20Data%20pre-processing.md)<br/>
2. [Implementation of Transformer](https://github.com/DolceLatte/NeuReduce_pytorch/blob/main/Model_summary.md)<br/>
3. [Implementation of LSTM Based on Attention](https://github.com/DolceLatte/NeuReduce_pytorch/blob/main/LSTM.md)<br/>

### Comparative evaluation result
|Method|Correct|Ratio|Result Length|Solving Time
|---|---|---|---|---|
|LSTM|*강조1*|테스트3|테스트3|테스트3|
|LSTM-Attention|**강조2**|테스트3|테스트3|테스트3|
|Transformer|8950|82.22|테스트3|0.026s|


2*(x&(y|z))+(~x|~z)-2*(y^z)+(~y^z)-2*(y^~z)-(x&y)
trg 		= -(~x&(y^z))
predicted trg 	= -(~x&(y^z))
