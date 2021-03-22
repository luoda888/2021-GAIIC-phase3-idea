# 2021-GAIIC-phase3-idea
非常荣幸能够拿到周星星，目前的分数是使用了stacking的效果，看到群里大佬们无私分享学到了很多，自己也想做一个稍微详细一些的分享，将目前思路抛砖引玉给大家，一起学习！！

## 个人拙见
### 数据层面
```
   q1-q2 = 1, q2-q3 = 1 ---> q1-q3 = 1
   q1-q2 = 1, q2-q3 = 0 ---> q1-q3 = 0
   构造强连通分量后大概增广了9000条数据，提升大概是2k左右
   随机负采样效果不佳(猜测是目前给的pair已经是区分难度较大的，导致随机负采样的样本过于简单，反而让较难样本的分类出现bias)
   
   使用数据对偶(q1q2 + q2q1)，取决于模型的效果，我的经验是Bert + 对偶没啥变化，Match / GBDT + 对偶是有显著提升的
   
   MAXLEN = 32, 这个长度已经几乎覆盖了所有的文本，主要是模型能训练的快很多...
```

### 思路1 : BERT (框架: PyTorch - Transformers)
```
   在nn这边可以分解为mask + pretrain + fine-tuning的组合(梯度累积=2 / FP16 / 训练数据是q1q2 + q2q1) (again = 从0训练, transfer = 加载预训练权重)
      1. Transfer NeZha-Large (normal MLM) (50EPOCH / LR5e-5 / LabelSmoothing0.01 / SEED 2021) (MLM Loss ≈ 0.15)
      2. Transfer NeZha-Large (ngram MLM) (100EPOCH / LR5e-5 / LabelSmoothing0 / SEED 2021) (MLM Loss ≈ 0.3)
      3. Transfer NeZha-Base (normal MLM) (50EPOCH / LR6e-5 / LabelSmoothing0 / SEED 2021) (MLM Loss ≈ 0.3)
      4. Again Bert-Base (normal MLM) (300EPOCH / LR5e-5 / LabelSmoothing0 / SEED 2021) (MLM Loss ≈ 0.1)
      5. Again Electra-Base (ngram MLM) (200EPOCH / LR1e-4 / LabelSmoothing0.01 / SEED 2021) (MLM Loss ≈ 0.1)
    Fine Tune （数据是q1q2, 不知道为啥对称会过拟合）
      线下结果发现复杂的模型结构接在Bert后边效果还不如CLS的... 目前尝试能够有用的
      1. 第一层
      2. 最后一层
      3. MeanPool
      4. 第一层 + 最后一层 + MeanPool
    大致效果 NeZha-Large(ngram) > NeZha-Large(normal) > NeZha-Base(normal) > Electra-Base(ngram MLM) > Bert-Base (normal MLM)
    最好的单模 NeZha-Large(ngram)(0.909) + FGM(3k) + Lookahead(1k) = 0.913 (5fold-offline = 0.977) (代码参考tutule大佬开源的nezha-torch预训练)
      分析一下为啥ngram会有用如：我喜欢你，拆分成字是，“我，喜，欢，你”。但是ngram能捕获一定的如 “喜欢” 这种词级别的信息，类似于全词遮罩的做法
```

### 思路2 : Match (框架: Tensorflow - Transformers)
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
    分类模型的话，普遍效果不太好，感觉是没有捕捉到q1/q2的语义差异, *表示线上未提交
        1. RCNN offline 0.93 / *
        2. CRNN offline 0.92 / *
        3. DeepMOJI offline 0.93 / *
```

### 思路3 : GBDT (框架: Xgboost & LightGBM & Catboost)
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

### MASK、数据构造方面
```
   0. 没有对词频做筛选，使用了全部的词
   1. ngram mask 在这起到的感觉是类似 脱敏前的 wwm mask的效果
   2. random mask的策略也比较重要，可以试试针对oov的mask
   3*. follow ngram mask的思路，实际上是构造了新词，那我们可以手动的生成各种ngram的词汇，从而进行数据增强
   4. 一定要使用q1 + " " + q2这种方式来预训练，比 np.hstack((q1, q2))要好很多
```

### 调参炼丹方面
```
   1. FGM (eps=0.5) > PGD(eps=0.5, alpha=0.6, K=3) (3-5k, 感觉是参数设置问题不太好收敛)
   2. Lookahead (1k, 很慢)
   3. LabelSmoothing (0.2k, 收敛变慢...)
   4. 字向量(直接训练出来的Embedding) + 动态词向量(字向量+LSTM) > Ori Embedding (3k, 0.878--->0.881)
   5. KFold 交叉验证的折数在0.91后收益减小 (3-5k, 单模单折我理解指全数据训练固定Epoch)
   6. Stacking使用的是lgb,大概放了10个模型进去(orz跑的太慢了)，对复赛是不友好的..还是得努力提升单模
```

### 打算实践的探索思路
```
   1. 对比学习A，Follow triplet-loss (q1, q2, q3, label), 可以使用大量的闭包关系来构造(author, left, right)数据，但是难点在测试集如何构造，以及图上的孤立点如何构造
   2. 对比学习B，Follow SCL (facebook & Standford)的方法，在pretrain的时候把同类(query group)的当正样本，不同类的当负样本，MLM+分类任务
   3. 对比学习C，Follow SimCLR，做二次的pretrain，具体定义数据表征方法为EDA(EMNLP 2019), x'与x''为正样本，x'与y'为负样本，吃机器，跑不动
   4. 半监督，可以从test里构造大量的pair来构造自标注样本，计算量有点大，笛卡尔积(100000 * 25000)，目前没想到优化方法，机器吃不消
   5. Link-Prediction，难点在图节点怎么定义
```
另，赛后想系统性的在公开数据集中复现文本匹配的各类模型、各类预训练策略的效果，有想法的朋友随时戳我呀~
