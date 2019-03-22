#預測鐵達尼號倖存者
import pandas as pd
import tensorflow as tf
import numpy as np
data=pd.read_csv("data/train.csv")

#male=1 femal=0
data['Sex']=data['Sex'].apply(lambda s:1 if s=="male" else 0)
#空值補0
data=data.fillna(0)
#取特定屬性
dataset_X=data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dataset_X=dataset_X.values
data['Deceased']=data['Survived'].apply(lambda s: int(not s))

#增加一個屬性Deceased
dataset_Y=data[['Deceased','Survived']]
dataset_Y=dataset_Y.values

#用train_test_split切割 20%為訓練資料
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

#placeholder 佔位元
x=tf.placeholder(tf.float32, shape=[None,6])
y=tf.placeholder(tf.float32, shape=[None,2])

#Variable 變數
W=tf.Variable(tf.random_normal([6,2]), name='weights')
b=tf.Variable(tf.zeros([2]), name='bias')

#logistic regression: y'=softmax(x*w+b)
logistic=tf.matmul(x,W)+b
y_pred=tf.nn.softmax(logistic)

#lossfunction: cross entropy
#1e-10=> 避免梯度爆炸
cross_entropy=-tf.reduce_sum(y*tf.log(y_pred+1e-10),reduction_indices=1)
cost=tf.reduce_mean(cross_entropy)

#training 使用GradientDescent最佳化 學習率0.001
train_op=tf.train.GradientDescentOptimizer(0.001).minimize(cost)

#session.run tf才開始運作
with tf.Session() as sess:
    #區塊名稱，graph裡面會標記
    with tf.name_scope('Run'):
        #初始化所有參數
        sess.run(tf.global_variables_initializer())

        #for loop for training
        for epoch in range(40):
            total_loss=0
            for i in range(len(X_train)):
                feed={x: [X_train[i]], y: [Y_train[i]]}
                #執行train_op的結果給_   cost結果給loss, feed_dict為placeholder需要值時的參考
                _, loss=sess.run([train_op, cost], feed_dict=feed)
                total_loss+=loss
            print('Epoch: %04d, total loss=%.9f' % (epoch+1, total_loss))
        print('training completely!')
        
        #執行y_pred 當運算過程遇到x 以X_train 代入
        pred=sess.run(y_pred, feed_dict={x:X_train})
        correct=np.equal(np.argmax(pred, 1), np.argmax(Y_train, 1))
        accuracy=np.mean(correct.astype(np.float32))
        print("Accuracy on validation set: %.9f" % accuracy)
        #儲存訓練結果
        saver=tf.train.Saver()
        saver.save(sess, 'variable/logisticmodel')

#test
test_data=pd.read_csv("data/test.csv")
test_data=test_data.fillna(0)
test_data['Sex']=test_data['Sex'].apply(lambda s:1 if s=='male' else 0)
X_test=test_data[['Sex','Age','Pclass','SibSp','Parch','Fare']]

#用summary 儲存結果給 tensorboard 用
#cmd =>>   tensorboard --logdir=PATH 執行完，網址localhost:6006開啟
train_writer=tf.summary.FileWriter('graph/tfboard_test', sess.graph)
train_writer.close()

with tf.Session() as sess:
    #載入變數
    saver.restore(sess, 'variable/logisticmodel')
    predictions=np.argmax(sess.run(y_pred, feed_dict={x: X_test}), 1)
    submission=pd.DataFrame({"PassengerId":test_data["PassengerId"],"Survived": predictions})
    submission.to_csv("answer.csv", index=False)