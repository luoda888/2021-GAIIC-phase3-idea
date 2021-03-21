# 2021-GAIIC-phase3-idea
非常荣幸能够拿到周星星，目前的分数是使用了stacking的效果，将目前思路抛砖引玉给大家！

## 个人拙见
### 思路1 : BERT
```
   在nn这边可以分解为mask + pretrain + fine-tuning的组合(梯度累积=2 / FP16 / 训练数据是q1q2 + q2q1)
      1. NeZha-Large (normal MLM) (50EPOCH / LR5e-5 / LabelSmoothing0.01 / SEED 2021) (MLM Loss ≈ 0.15)
      2. NeZha-Large (ngram MLM) (100EPOCH / LR5e-5 / LabelSmoothing0 / SEED 2021) (MLM Loss ≈ 0.3)
      3. NeZha-Base (normal MLM) (50EPOCH / LR6e-5 / LabelSmoothing0 / SEED 2021) (MLM Loss ≈ 0.3)
      4. Bert-Base (normal MLM) (300EPOCH / LR5e-5 / LabelSmoothing0 / SEED 2021) (MLM Loss ≈ 0.1)
      5. Electra-Base (ngram MLM) (200EPOCH / LR1e-4 / LabelSmoothing0.01 / SEED 2021) (MLM Loss ≈ 0.1)
    Fine Tune （数据是q1q2, 不知道为啥对称会过拟合）
      线下结果发现复杂的模型结构接在Bert后边效果还不如CLS的... 目前尝试的
      1. 第一层
      2. 最后一层
      3. MeanPool
      4. 第一层 + 最后一层 + MeanPool
    大致效果 NeZha-Large(ngram) > NeZha-Large(normal) > NeZha-Base(normal) > Electra-Base(ngram MLM) > Bert-Base (normal MLM)
      分析一下为啥ngram会有用如：我喜欢你，拆分成字是，“我，喜，欢，你”。但是ngram能捕获一定的如 “喜欢” 这种词级别的信息，类似于全词遮罩的做法
```

### 思路2 : Match
```
    训练的数据 q1q2 + q2q1 (对偶)
    Match可以用到传统的文本匹配模型，文本分类模型，分享几个有用的trick。
        1. Embedding后接Dropout
        2. 对LSTM加TimeSeries(Dense)
        3. window=1的FastText有一定能力的纠错功能（参考夕小瑶的卖萌屋的回答）(具体使用可以拼接、加权，线下看是拼接更好)
        4. Glove + FastText(window=1) + Word2Vec 效果好于单独使用词向量
    分享几个有用的模型-匹配 (赛后会将keras的实现版本开源) (下面的分数是 传统Embedding / BERT静态向量, *表示没跑)
        1. ESIM 0.883 / 0.895（注意，很多开源库实现里，在soft_attention后，没有将left, right做交互，只对left和right分别做拼接，将这个修正后分数可以从84-88）
        2. DIIN 0.885 / 0.901
        3. DRCN 0.87 / *
        4. CAFE 0.85 / *
        5. BiMPM 0.84 / *
        6. RE2 0.82 / * (复现的有问题)
        7. ERCNN 0.85 / * (还是感觉复现的有问题)
    分类模型的话，普遍效果不太好，感觉是没有捕捉到q1/q2的语义差异
        1. RCNN offline 0.93 / *
        2. CRNN offline 0.92 / *
        3. DeepMOJI offline 0.93 / *
```

### 思路3 ： GBDT
```
   训练的数据 q1q2 + q2q1
   类似的比赛里GBDT类特征效果还是挺不错的，这次使用的特征是缝合怪...把paipaidai Top1/Top14, Quora Pair的开源代码copy过来了..
   分5个filed:
      1. 图特征： pagerank类(paipaidai Top1)，hash_subgraph_qdiff(paipaidai Top1)
      2. 统计类特征： 词转化率，各种距离的统计(Quora Pair)
      3. 主题类特征： TFIDF + NMF, TFIDF + LSI
      4. 交互类特征： 将q1 q2的各种不同类型(meanpool / maxpool / first / last)的向量(bert / w2v / tfidf-svd)进行统计，计算dot, cosine等
      5. 学习类特征： 将q1 q2拼接成一个句子，使用tfidf + countvec拼接当特征，过各种弱学习器后拼接概率
   对GBDT类模型，校验对称性的方法就是，预测q1 q2和q2 q1的概率，看这两组概率的相关性，如果较低说明有必要对偶一下（提升会比较明显）
```
