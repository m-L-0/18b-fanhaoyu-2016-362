作者：范浩宇

算法：谱聚类

代码：python

数据处理：归一化

数据集：鸢尾花数据集




```python
import os
import scipy.io as scio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pandas
#加载数据，划分训练集，验证集
minMax = MinMaxScaler()
data_dict = {}
data_path = "./data"
for i in os.listdir(data_path):
    if i != 'test.mat':
        data = scio.loadmat(os.path.join(data_path,i))
        data_dict[i.split(".")[0]] = data[i.split(".")[0]]
        
for i in data_dict.keys():
    data_dict[i] = minMax.fit_transform(data_dict[i])
X = []
for i in data_dict.keys():
    for j in data_dict[i]:
        X.append(j)
X = np.array(X)
X = Normalizer().fit_transform(X)

count = 0
label = 0
for i in data_dict.keys():
    for j in range(len(data_dict[i])):
        Y.append(label)
        count += 1
    label += 1
Y = np.array(Y)
idx = np.random.randint(0,6924,6924)
X,Y = X[idx],Y[idx]

#初始化模型，训练模型
clf = SVC()
clf.fit(X,Y)
y_pred = clf.predict(X)
valid = (Y == y_pred)


#预测测试集的结果
test_path = './data/test.mat'
test_data = scio.loadmat(test_path)['data_test_final']
test = minMax.fit_transform(test_data)
label = clf.predict(test)
x = pandas.DataFrame(label)
x.to_csv('results.csv')
```
