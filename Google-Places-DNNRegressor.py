import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from skmultilearn.model_selection import iterative_train_test_split
from tensorflow.keras.layers import *
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)
print('x' in np.arange(5))
tf.get_logger().setLevel(logging.ERROR)

df = pd.read_csv('/home/gerard/2020-11-15_00:55:38_barcelona-supermercado.csv', delimiter=';')
df = df.drop('address', axis=1).drop('errors', axis=1)

keywords = ['Supermercado', 'Tienda de alim', 'ultramarinos', 'delicatessen', 'Fruter', 'congelados', 'Mercado']

df_filter = df.type.isin(keywords)
df = df[df_filter]

df = df[df.comments < 500]

df.fillna(value=0, inplace=True)

X = df[['rate', 'comments']].values
y = df[['lat', 'lon']].values

X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.3)

def train_input_fn():
    X_train_dic = {}
    X_train_dic['rate'] = []
    X_train_dic['comments'] = []
    count = 0

    for test in np.nditer(X_train):
        if count % 2 == 0:
            X_train_dic['rate'].append(float(test))
        else:
            X_train_dic['comments'].append(float(test))
        count += 1

    return X_train_dic, y_train


def pred_input_fn():
    X_test_dic = {}
    X_test_dic['rate'] = []
    X_test_dic['comments'] = []
    count = 0

    for test in np.nditer(X_test):
        if count % 2 == 0:
            X_test_dic['rate'].append(float(test))
        else:
            X_test_dic['comments'].append(float(test))
        count += 1

    return X_test_dic


rate = tf.feature_column.numeric_column("rate",shape=[1])
comments = tf.feature_column.numeric_column("comments",shape=[1])

model = tf.estimator.DNNRegressor(
  hidden_units    = [1024, 512, 256, 128],
  feature_columns = [rate, comments],
  label_dimension = y_train.shape[1],
  activation_fn   = 'relu',
  dropout         = 0.2,
  optimizer       = 'Adam',
  model_dir       = './Google-Places-DNNRegressor'
  )

mode = int(input('''
Choose one:
1. Training
2. Predicting
'''))


if mode == 1:
    print('Training...')
    for _ in range(100):
      r = model.evaluate(train_input_fn,steps=1);
      print('Loss:',r['loss'])
      model.train(train_input_fn,steps=1000)

elif mode == 2:
    predictions = model.predict(pred_input_fn)
    pred_output = {}
    pred_output['latitude'] = []
    pred_output['longitude'] = []

    for test in range(y_test.shape[0]):
        # print(next(predictions)['predictions'][:])
        pred_output['latitude'].append(next(predictions)['predictions'][0])
        pred_output['longitude'].append(next(predictions)['predictions'][1])

    scatter_q = input('''Do you want a scatterplot of the results? y/n \n''')

    if scatter_q == 'y':
        test_predictions = pd.DataFrame.from_dict(pred_output)
        test_predictions['type'] = 'model_prediction'
        y_test_df = pd.DataFrame(y_test, columns=['latitude', 'longitude'])
        y_test_df['type'] = 'y_test_true'
        frames = [y_test_df, test_predictions]
        pred_df = pd.concat(frames)

        pred_plot = sns.scatterplot(x='latitude', y='longitude', hue='type', data=pred_df, alpha=0.6, linewidth=0, palette='viridis')
        fig = pred_plot.get_figure()
        fig.savefig('prediction_scatterplot.png', dpi = 600)
    else:
        exit()

else:
    exit()
