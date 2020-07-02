# 王宇彬 3220190887
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from pyod.models.abod import ABOD
import numpy as np
data = pd.read_csv("D:/dataMiner/abalone/abalone/meta_data/meta_abalone.csv")
rates = data['anomaly.rate'].values.reshape(-1)
name = "abalone_benchmark_"
scaler = MinMaxScaler(feature_range=(0,1))
# 保存准确率
results = [] 
fig = plt.figure(figsize=(20,20))
for i in range(1,10):
    outliers_fraction = rates[i - 1]
    if(outliers_fraction > 0.5):
        outliers_fraction = 0.5
    id = str(i)
    length = len(id)
    for j in range(4 - length):
        id = '0' + id
    file = "D:/dataMiner/abalone/abalone/benchmarks/" + name + id + ".csv"
    if not os.path.exists(file):
        continue
    print(file)
    df = pd.read_csv(file)
    # 将表格中的标签列属性置为0和1
    df.loc[df['ground.truth']=='anomaly','ground.truth'] = 1 
    df.loc[df['ground.truth']=='nominal','ground.truth'] = 0
    y = df['ground.truth'].values.reshape(-1)
    df[['V1','V2','V3','V4','V5','V6','V7']] = scaler.fit_transform(df[['V1','V2','V3','V4','V5','V6','V7']])
    x1 = df['V1'].values.reshape(-1,1)
    x2 = df['V2'].values.reshape(-1,1)
    x3 = df['V3'].values.reshape(-1,1)
    x4 = df['V4'].values.reshape(-1,1)
    x5 = df['V5'].values.reshape(-1,1)
    x6 = df['V6'].values.reshape(-1,1)
    x7 = df['V7'].values.reshape(-1,1)
    x = np.concatenate((x1,x2,x3,x4,x5,x6,x7),axis=1)
    abod = ABOD(contamination=outliers_fraction)
    abod.fit(x)
    y_pred = abod.predict(x)
    fpr,tpr,threshold = roc_curve(y,y_pred) ###计算真阳性率和假阳性率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    lw = 2
    ax = fig.add_subplot(3, 3, i)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    sum = 0
    # 计算正确率
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            sum += 1
    results.append(sum / len(y))
plt.show()
# 展示abalone前9个数据集用ABOD方法得到的ROC曲线



plt.figure(figsize=(18,8))
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("数据集",fontdict={'weight':'normal','size': 20})
plt.ylabel("准确率",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13) 
x = ['1','2','3','4','5','6','7','8','9']
plt.bar(x,results)  
plt.title('各个数据集上的离群点判别准确率',fontdict={'weight':'normal','size': 20})
plt.show()
# 展示abalone前9个数据集用ABOD方法得到的准确率


from pyod.models.hbos import HBOS
results = []
random_state = np.random.RandomState(42)
fig = plt.figure(figsize=(20,20))
for i in range(1,10):
    outliers_fraction = rates[i - 1]
    if(outliers_fraction > 0.5):
        outliers_fraction = 0.5
    id = str(i)
    length = len(id)
    for j in range(4 - length):
        id = '0' + id
    file = "D:/dataMiner/abalone/abalone/benchmarks/" + name + id + ".csv"
    if not os.path.exists(file):
        continue
    print(file)
    df = pd.read_csv(file)
    df.loc[df['ground.truth']=='anomaly','ground.truth'] = 1
    df.loc[df['ground.truth']=='nominal','ground.truth'] = 0
    y = df['ground.truth'].values.reshape(-1)
    df[['V1','V2','V3','V4','V5','V6','V7']] = scaler.fit_transform(df[['V1','V2','V3','V4','V5','V6','V7']])
    x1 = df['V1'].values.reshape(-1,1)
    x2 = df['V2'].values.reshape(-1,1)
    x3 = df['V3'].values.reshape(-1,1)
    x4 = df['V4'].values.reshape(-1,1)
    x5 = df['V5'].values.reshape(-1,1)
    x6 = df['V6'].values.reshape(-1,1)
    x7 = df['V7'].values.reshape(-1,1)
    x = np.concatenate((x1,x2,x3,x4,x5,x6,x7),axis=1)
    hbos = HBOS(contamination=outliers_fraction)
    hbos.fit(x)
    y_pred = hbos.predict(x)
    fpr,tpr,threshold = roc_curve(y,y_pred) ###计算真阳性率和假阳性率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    lw = 2
    ax = fig.add_subplot(3, 3, i)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    sum = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            sum += 1
    results.append(sum / len(y))
plt.show()
# 展示abalone前9个数据集用HBOS方法得到的ROC曲线


plt.figure(figsize=(18,8))
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("数据集",fontdict={'weight':'normal','size': 20})
plt.ylabel("准确率",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13) 
x = ['1','2','3','4','5','6','7','8','9']
plt.bar(x,results)  
plt.title('各个数据集上的离群点判别准确率',fontdict={'weight':'normal','size': 20})
plt.show()
# 展示abalone前9个数据集用ABOD方法得到的准确率


data = pd.read_csv("D:/dataMiner/wine/wine/meta_data/meta_wine.csv")
rates = data['anomaly.rate'].values.reshape(-1)
name = "wine_benchmark_"
scaler = MinMaxScaler(feature_range=(0,1))
results = []
fig = plt.figure(figsize=(20,20))
for i in range(1,10):
    outliers_fraction = rates[i - 1]
    if(outliers_fraction > 0.5):
        outliers_fraction = 0.5
    id = str(i)
    length = len(id)
    for j in range(4 - length):
        id = '0' + id
    file = "D:/dataMiner/wine/wine/benchmarks/" + name + id + ".csv"
    if not os.path.exists(file):
        continue
    print(file)
    df = pd.read_csv(file)
    df.loc[df['ground.truth']=='anomaly','ground.truth'] = 1
    df.loc[df['ground.truth']=='nominal','ground.truth'] = 0
    y = df['ground.truth'].values.reshape(-1)
    df[['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']] \
    = scaler.fit_transform(df[['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']] )
    x1 = df['fixed.acidity'].values.reshape(-1,1)
    x2 = df['volatile.acidity'].values.reshape(-1,1)
    x3 = df['citric.acid'].values.reshape(-1,1)
    x4 = df['residual.sugar'].values.reshape(-1,1)
    x5 = df['chlorides'].values.reshape(-1,1)
    x6 = df['free.sulfur.dioxide'].values.reshape(-1,1)
    x7 = df['total.sulfur.dioxide'].values.reshape(-1,1)
    x8 = df['density'].values.reshape(-1,1)
    x9 = df['pH'].values.reshape(-1,1)
    x10 = df['sulphates'].values.reshape(-1,1)
    x11 = df['alcohol'].values.reshape(-1,1)   
    x = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11),axis=1)
    knn = KNN(contamination=outliers_fraction)
    knn.fit(x)
    y_pred = knn.predict(x)
    fpr,tpr,threshold = roc_curve(y,y_pred) ###计算真阳性率和假阳性率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    lw = 2
    ax = fig.add_subplot(3, 3, i)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    
    sum = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            sum += 1
    results.append(sum / len(y))
plt.show()
# 展示wine前9个数据集用knn方法得到的ROC曲线


plt.figure(figsize=(18,8))
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("数据集",fontdict={'weight':'normal','size': 20})
plt.ylabel("准确率",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13) 
x = ['1','2','3','4','5','6','7','8','9']
plt.bar(x,results)  
plt.title('各个数据集上的离群点判别准确率',fontdict={'weight':'normal','size': 20})
plt.show()
# 展示wine前9个数据集用knn方法得到的准确率



results = []
fig = plt.figure(figsize=(20,20))
for i in range(1,10):
    outliers_fraction = rates[i - 1]
    if(outliers_fraction > 0.5):
        outliers_fraction = 0.5
    id = str(i)
    length = len(id)
    for j in range(4 - length):
        id = '0' + id
    file = "D:/dataMiner/wine/wine/benchmarks/" + name + id + ".csv"
    if not os.path.exists(file):
        continue
    print(file)
    df = pd.read_csv(file)
    df.loc[df['ground.truth']=='anomaly','ground.truth'] = 1
    df.loc[df['ground.truth']=='nominal','ground.truth'] = 0
    y = df['ground.truth'].values.reshape(-1)
    df[['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']] \
    = scaler.fit_transform(df[['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']] )
    x1 = df['fixed.acidity'].values.reshape(-1,1)
    x2 = df['volatile.acidity'].values.reshape(-1,1)
    x3 = df['citric.acid'].values.reshape(-1,1)
    x4 = df['residual.sugar'].values.reshape(-1,1)
    x5 = df['chlorides'].values.reshape(-1,1)
    x6 = df['free.sulfur.dioxide'].values.reshape(-1,1)
    x7 = df['total.sulfur.dioxide'].values.reshape(-1,1)
    x8 = df['density'].values.reshape(-1,1)
    x9 = df['pH'].values.reshape(-1,1)
    x10 = df['sulphates'].values.reshape(-1,1)
    x11 = df['alcohol'].values.reshape(-1,1)   
    x = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11),axis=1)
    hbos = HBOS(contamination=outliers_fraction)
    hbos.fit(x)
    y_pred = hbos.predict(x)
    fpr,tpr,threshold = roc_curve(y,y_pred) ###计算真阳性率和假阳性率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    lw = 2
    ax = fig.add_subplot(3, 3, i)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    sum = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            sum += 1
    results.append(sum / len(y))
plt.show()
# 展示wine前9个数据集用HBOS方法得到的ROC曲线


plt.figure(figsize=(18,8))
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("数据集",fontdict={'weight':'normal','size': 20})
plt.ylabel("准确率",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13) 
x = ['1','2','3','4','5','6','7','8','9']
plt.bar(x,results)  
plt.title('各个数据集上的离群点判别准确率',fontdict={'weight':'normal','size': 20})
plt.show()
# 展示wine前9个数据集用knn方法得到的准确率


# 本报告对wine_benchmark和abalone_benchmark进行离群点的检测。由于这两个数据集都包含1800多个表格，所以仅对前9个数据集进行可视化展示。
# 对表格的处理，首先是读取表格，把表格中的属性进行归一化，然后提取出来作为特征向量。对于标签属性，则转化为0和1，0代表正常点，1代表离群点。
# 对abalone_benchmark数据集先用ABOD方法进行离群点检测，发现ROC区域面积可以达到0.7，准确率也能到70%；而用HBOS方法进行检测，效果则差了许多。
# 对wine_benchmark数据集采用KNN和HBOS两种方法进行检测，发现准确率都是略大于0.5。ROC面积也是在0.5左右。
# 离群点检测有许多种方法，对于不同的数据集，要找到合适的方法。