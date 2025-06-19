# Word2Vec

Word2Vec是一种流行的词嵌入（Word Embedding）技术，由Tomas Mikolov等人在2013年提出。它是一种基于神经网络NNLM的语言模型，旨在通过学习词与词之间的上下文关系来生成词的密集向量表示。Word2Vec的核心思想是利用词在文本中的上下文信息来捕捉词之间的语义关系，从而使得语义相似或相关的词在向量空间中距离较近。
```python
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```
Word2Vec模型主要有两种架构：连续词袋模型CBOW(Continuous Bag of Words)是根据目标词上下文中的词对应的词向量, 计算并输出目标词的向量表示；Skip-Gram模型与CBOW模型相反, 是利用目标词的向量表示计算上下文中的词向量. 实践验证CBOW适用于小型数据集, 而Skip-Gram在大型语料中表现更好。
- CBOW（Continuous Bag of Words）：根据上下文词预测中心词
- Skip-gram：根据中心词预测上下文词
  

> **"The cat is sleeping on the sofa."**

#### CBOW的训练样本：
我们设定窗口大小为 3（表示左右各三个词为上下文）。
* 目标：预测中心词 "sleeping"
* 上下文词是：["The", "cat", "is", "on", "the", "sofa"]
  
#### Skip-gram 的训练样本：
* 目标：用中心词"sleeping"预测它的上下文词。
* 上下文词是：["The", "cat", "is", "on", "the", "sofa"]
* 会生成"sleeping"分别和上下文词的训练对


相比于传统的高维稀疏表示（如One-Hot编码），Word2Vec生成的是低维（通常几百维）的密集向量，有助于减少计算复杂度和存储需求。Word2Vec模型可以很好地泛化到训练中未出现过的词，因为它是基于上下文信息学习的，而不是基于词典。但由于CBOW/Skip-Gram模型是基于局部上下文的，无法捕捉到长距离的依赖关系，缺乏整体的词与词之间的关系，因此在一些复杂的语义任务上表现不佳。

*Question：OOV的泛化能力*

Gensim 是一个非常流行的 Python 工具库，用于训练 Word2Vec模型，它支持两种模型：CBOW 和 Skip-gram，可以通过设置参数`sg`来选择：`sg=0（default）`是CBOW， `sg=1`是Skip-gram


```python
from gensim.models import Word2Vec


sentences = [
    ["the", "cat", "is", "sleeping", "on", "the", "sofa"],
    ["a", "mouse", "runs", "across", "the", "floor"],
    ["she", "opens", "one", "eye"],
    ["then", "goes", "back", "to", "sleep"]
]
model = Word2Vec(sentences, vector_size=100, window=8, min_count=1)
model.wv.most_similar(positive=["cat"])
```




    [('mouse', 0.19911441206932068),
     ('floor', 0.17272186279296875),
     ('sleeping', 0.17018885910511017),
     ('back', 0.1528114527463913),
     ('sofa', 0.14595060050487518),
     ('on', 0.06408978253602982),
     ('she', 0.04652618616819382),
     ('goes', 0.0014541522832587361),
     ('a', -0.0027540235314518213),
     ('is', -0.013514946214854717)]


