作者：范浩宇

算法：KNN 

代码：python,tensorflow

数据处理：归一化

数据集：鸢尾花数据集

#归一化数据
X = np.array([[ (x-min(data))/(max(data)-min(data)) for x in data  ]   for data in X])

#确定k值
k = 10

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,random_state=1)
# 输入占位符
X_train = tf.placeholder("float", [None, 4])
X_test = tf.placeholder("float", [4])


distance = tf.reduce_sum(tf.abs(tf.add(X_train, tf.negative(X_test))), reduction_indices=1)
#计数正确预测的个数
correct = 0
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 开始测试
    for i in range(len(Xtest)):
        # 获取当前样本与各样本之间的距离
        dis = sess.run(distance,feed_dict={X_train: Xtrain, X_test: Xtest[i, :]})
        # 获取最近的k个样本
        indexs = np.argsort(dis)[:k]
        candidate_Y = Ytrain[indexs]
        #投票表决
        pred = np.argmax(np.bincount(candidate_Y))
        print("Test", i, "Prediction:", yy,"True Class:", Ytest[i])
        # 如果分类正确，correct += 1
        if Ytest[i] == pred:
            correct += 1
    acc = correct/len(Ytest)
    print("Acc:","{}%".format(acc * 100))
