2016-07-05CSDN大数据

# Dropout介绍
dropout技术是神经网络和深度学习模型的一种简单而有效的正则化方式。本文将向你介绍dropout正则化技术，并且教你如何在Keras中用Python将其应用于你的模型。读完本文之后，你将了解：
  * dropout正则化的原理
  * 如何在输入层使用dropout
  * 如何在隐藏层使用dropout
  * 如何针对具体问题对dropout调优

# 神经网络的Dropout正则化
Dropout是Srivastava等人在2014年的一篇论文中提出的一种针对神经网络模型的正则化方法 Dropout: A Simple Way to Prevent Neural Networks from Overfitting。

Dropout的做法是在训练过程中随机地忽略一些神经元。这些神经元被随机地“抛弃”了。也就是说它们在正向传播过程中对于下游神经元的贡献效果暂时消失了，反向传播时该神经元也不会有任何权重的更新。

随着神经网络模型不断地学习，神经元的权值会与整个网络的上下文相匹配。神经元的权重针对某些特征进行调优，具有一些特殊化。周围的神经元则会依赖于这种特殊化，如果过于特殊化，模型会因为对训练数据过拟合而变得脆弱不堪。神经元在训练过程中的这种依赖于上下文的现象被称为复杂的协同适应（complex co-adaptations）。

你可以想象一下，如果在训练过程中随机丢弃网络的一部分，那么其它神经元将不得不介入，替代缺失神经元的那部分表征，为预测结果提供信息。人们认为这样网络模型可以学到多种相互独立的内部表征。

这么做的效果就是，网络模型对神经元特定的权重不那么敏感。这反过来又提升了模型的泛化能力，不容易对训练数据过拟合。

# Keras的Dropout 正则化
Dropout的实现很简单，在每轮权重更新时随机选择一定比例（比如20%）的节点抛弃。Keras的Dropout也是这么实现的。Dropout技术只在模型训练的阶段使用，在评估模型性能的时候不需使用。

接下来我们看看Dropout在Keras中的一些不同用法。本例子使用了声呐数据集（Sonar dataset）。这是一个二分类问题，目的是根据声呐的回声来正确地区分岩石和矿区。这个数据集非常适合神经网络模型，因为所有的输入都是数值型的，且具有相同的量纲。

数据集可以从UCI机器学习代码库下载。然后把声呐数据集放在当前工作路径下，文件命名为sonar.csv。

我们会用scikit-learn来评价模型质量，为了更好地挑拣出结果的差异，采用了十折交叉验证（10-fold cross validation）方法。

每条数据有60个输入值和1个输出值，输入值在送入模型前做了归一化。基准的神经网络模型有两个隐藏层，第一层有60个节点，第二层有30个。使用了随机梯度下降的方法来训练模型，选用了较小的学习率和冲量。

完整的基准模型代码如下所示:

```python

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# baseline
def create_baseline():
# create model
model = Sequential()
model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
 model.add(Dense(30, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))
# Compile model
sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

运行代码，分类的准确率大概为82%。

# 在可见层使用Dropout
Dropout可用于输入神经元，即可见层。

在下面这个例子里，我们在输入（可见层）和第一个隐藏层之间加入一层Dropout。丢弃率设为20%，就是说每轮迭代时每五个输入值就会被随机抛弃一个。

另外，正如Dropout那篇论文中所推荐的，每个隐藏层的权重值都做了限制，确保权重范数的最大值不超过3。在构建模型层的时候，可以通过设置Dense Class的W_constraint参数实现。

学习率提高了一个数量级，冲量增加到0.9。这也是那篇Dropout论文的原文中所推荐的做法。

顺着上面基准模型的例子，下面的代码是包含输入层dropout的网络模型。
