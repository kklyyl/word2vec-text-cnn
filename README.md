# Text classification with CNN and Word2vec

### 3.0训练记录：
```
Epoch: 15
step: 2400,train loss: 0.677, train accuracy: 0.828, val loss: 1.886, val accuracy: 0.507 *
```
```
epoch=30 测试出来为了，防止过拟合epoch不应超过10；
```

### 2.0训练记录
#### 第1次训练:
```
train acc 0.48；val acc 0.37；test acc 0.35
数据欠拟合
```
#### 第2次训练:
```
train acc 0.45；val acc 0.27；test acc 0.25
更改条件：1.加入单个文字；2.加入标点符号和数字；3.x_text维度降为300；4. 词典大小总词汇改为25000
```
#### 第3次训练:
```
train acc 0.89；val acc 0.35；test acc 0.36
更改条件：1.Epoch更改为15轮；2.去掉了标点符号；
```
#### 第4次训练:
```
train acc 0.96；val acc 0.43；test acc 0.43
更改条件：1.Epoch更改为30轮；2.x_text维度降为300降为256 3.predict时把vocab_size改成词典大小23051；
实验效果：20轮训练以后 val acc 逐渐趋紧0.44-0.46左右；val acc上升速度很慢；随机抽取十个数据并不能达到40%的准确度
```
#### 第5次训练:
```
数据量拓展一倍以后的训练
step: 1600,train loss: 1.245, train accuracy: 0.625, val loss: 1.982, val accuracy: 0.474*
```

# 原始模型训练记录（字符集文本表示+textcnn+textrnn）
```
cnn: train acc 0.64；val acc 0.57；test acc 0.56
rnn: train acc 0.47；val acc 0.37；test acc 0.37 （epoch:10 实际5）
rnn: train acc 0.79；val acc 0.40；test acc 0.40 （epoch:20 实际10）
rnn比CNN模型要慢几倍到十几倍。因为，后一个时间步的输出依赖于前一个时间步的输出，无法进行并行处理，导致模型训练的速度慢，且准确度较低，所以实验舍弃使用rnn模型。
```

#### 第2次训练:
```
cnn 词汇表以词力度划分，导致模型过拟合
Iter:    900, Train Loss:   0.67, Train Acc:  79.69%, Val Loss:    1.9, Val Acc:  53.96%, Time: 0:03:09 *
```