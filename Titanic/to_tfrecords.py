import pandas as pd
import tensorflow as tf
def transform_to_tfrecords():
    data=pd.read_csv('data/train.csv')
    tfrecord_file='train.tfrecords'

    def int_features(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def float_features(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    writer=tf.python_io.TFRecordWriter(tfrecord_file)
    for i in range(len(data)):
        features= tf.train.Features(feature={
            'Age': float_features(data['Age'][i]),
            'Survived': int_features(data['Survived'][i]),
            'Pclass': int_features(data['Pclass'][i]),
            'Parch': int_features(data['Parch'][i]),
            'SibSp': int_features(data['SibSp'][i]),
            'Sex': int_features(1 if data['Sex'][i]=='male' else 0),
            'Fare': float_features(data['Fare'][i])
        })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
