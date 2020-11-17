import pandas as pd
import seaborn as sns
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from tensorflow.keras.models import load_model

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

model = load_model('google_places_model.h5')


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

for rate, comments in zip(X_test_dic['rate'], X_test_dic['comments']):
    print(rate, comments)
    prediction = model.predict(np.array([[rate, comments]]))
    print(prediction)

test_predictions = model.predict(X_test)
test_predictions = pd.DataFrame(test_predictions.reshape(X_test.shape), columns=['latitude', 'longitude'])
test_predictions['type'] = 'model_prediction'
y_test_df = pd.DataFrame(y_test, columns=['latitude', 'longitude'])
y_test_df['type'] = 'y_test_true'
frames = [y_test_df, test_predictions]
pred_df = pd.concat(frames)

pred_plot = sns.scatterplot(x='latitude', y='longitude', hue='type', data=pred_df, alpha=0.6, linewidth=0, palette='viridis')
fig = pred_plot.get_figure()
fig.savefig('prediction_scatterplot.png', dpi = 600)
