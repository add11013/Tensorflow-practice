import os 
import numpy as np
import pandas as pd 
import tensorflow as tf
import tflearn
train_data= pd.read_csv('data/train.csv')
X=train_data[['Sex','Age','Pclass','SibSp','Parch','Fare']].values
train_data['Deceased']=train_data['Survived'].apply(lambda s:1 if s==1 else 0)
Y=train_data[['Deceased','Survived']].values

#arguments that can be set in command line
tf.app.flags.DEFINE_integer('epochs', 10, 'Training epochs')

#build the directory
ckpt_dir='./ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

#define classifier
n_features= X.shape[1]
input=tflearn.input_data([None, n_features])
y_pred=tflearn.layers.fully_connected(network, 2, activation='softmax')
net=tflearn.regression(y_pred)
model = tflearn.DNN(net)

# load the model save
if os.path.isfile(os.path.join(ckpt_dir, 'model.ckpt')):
    model.load(os.path.join(ckpt_dir, 'model.ckpt'))

#training
model.fit(X,Y, validation_set=0.1, n_epoch=tf.app.flags.FLAGS)
model.save(os.path.join(ckpt_dir, 'model.ckpt'))
metric= model.evaluate(X, Y)
print('Accuracy on train set: %.9f' % metric[0])

