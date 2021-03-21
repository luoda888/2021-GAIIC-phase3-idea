# 2021-GAIIC-phase3-idea
`  `非常荣幸能够拿到第二周的周星星，目前的分数是使用了stacking的效果，将目前思路抛砖引玉给大家！

## 个人拙见

```
    思路一：在nn这边可以分解为mask + pretrain + fine-tuning的组合(梯度累积=2 / FP16 / 训练数据是q1q2 + q2q1)
      1. NeZha-Large (normal MLM) (50EPOCH / LR5e-5 / LabelSmoothing0.01 / SEED 2021) (MLM Loss ≈ 0.15)
      2. NeZha-Large (ngram MLM) (100EPOCH / LR5e-5 / LabelSmoothing0 / SEED 2021) (MLM Loss ≈ 0.3)
      3. NeZha-Base (normal MLM) (50EPOCH / LR6e-5 / LabelSmoothing0 / SEED 2021) (MLM Loss ≈ 0.3)
      4. Bert-Base (normal MLM) (300EPOCH / LR5e-5 / LabelSmoothing0 / SEED 2021) (MLM Loss ≈ 0.1)
      5. Electra-Base (ngram MLM) (200EPOCH / LR1e-4 / LabelSmoothing0.01 / SEED 2021) (MLM Loss ≈ 0.1)
    Fine Tune
      线下结果发现复杂的模型结构接在Bert后边效果还不如CLS的... 目前尝试的
      1. 第一层
      2. 最后一层
      3. MeanPool
      4. 第一层 + 最后一层 + MeanPool
    大致效果 NeZha-Large(ngram) > NeZha-Large(normal) > NeZha-Base(normal) > Electra-Base(ngram MLM) > Bert-Base (normal MLM)
      分析一下为啥ngram会有用如：我喜欢你，拆分成字是，“我，喜，欢，你”。但是ngram能捕获一定的如 “喜欢” 这种词级别的信息，类似于全词遮罩的做法
```
