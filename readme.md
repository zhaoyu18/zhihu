# 知乎看山杯
比赛链接：https://biendata.com/competition/zhihu/

## 数据预处理
使用了数据集中词语级别的问题 title 和问题描述的信息以及词语级别的 embedding 向量。

将词语级别的问题 title 和问题描述拼接在一起，取前200个词作为训练数据，生成 pad sequences 和 embedding matrix。

## 模型
使用了两种 CNN 模型，通过使用不同的 kernel size 来建模不同距离的关系。

模型1：训练的问题数据通过 embedding 层后，分别输入不同 kernel size（1~8）的卷积层；每个卷积层输出经过 GlobalAveragePooling1D 池化层压缩；然后输入一个大小为1500的 Dense 层；最后通过大小为1999，activation 为 softmax 的 Dense 层输出对不同标签的预测结果；取 top 5 作为预测标签。5 fold 单模型得分为 0.406+。

模型2：和模型1结构基本一致，只是将 embedding 层 trainable 设置为 true。5 fold 单模型得分为 0.413+。

## Ensemble
将两个模型预测结果简单求和，然后取 top 5 标签作为最终预测结果，得分为 0.416+。

## 运行环境
Keras 2.0.3, Tensorflow 1.0.0, Windows 10

## 代码说明
generate_data.py 生成用于输入模型的 train/test pad sequences 和 embedding matrix。

generate_labels.py 生成用于训练模型的 label 数据，维度为1999（全部标签数）。1 表示问题属于该标签，0 反之。

model\_cnn\_0.py 训练模型1

model\_cnn\_1.py 训练模型2

make_submission.py 生成提交文件，取 top 5 的标签作为预测标签

utils.py 小工具

