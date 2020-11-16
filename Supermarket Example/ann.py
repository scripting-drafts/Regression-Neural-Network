import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('/home/gerard/2020-11-15_00:55:38_barcelona-supermercado.csv', delimiter=';')
df = df.drop('address', axis=1).drop('errors', axis=1)

keywords = ['Supermercado', 'Tienda de alim', 'ultramarinos', 'delicatessen', 'Fruter', 'congelados', 'Mercado']

df_filter = df.type.isin(keywords)
df = df[df_filter]

df = df[df.comments < 500]

df.fillna(value=0, inplace=True)

X = df[['rate', 'comments']].values
y = df[['lat', 'lon']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print('X_train shape: ' + str(X_train.shape))
# print('X_test shape: ' + str(X_test.shape))

model = Sequential()

model.add(Dense(2,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(2))
model.compile(optimizer='adam',loss='mse')

#early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=6000)#, callbacks=[early_stop])

# model.save('google_places_model.h5')
# later_model = load_model('google_places_model.h5')

# test_predictions = model.predict(X_test)
# test_predictions = pd.DataFrame(test_predictions.reshape(X_test.shape), columns=['model_predict_lat', 'model_predict_lon'])
# y_test_df = pd.DataFrame(y_test, columns=['y_test_lat', 'y_test_lon'])
# pred_df = pd.concat([y_test_df, test_predictions], axis=1)
# print(pred_df)

test_predictions = model.predict(X_test)
test_predictions = pd.DataFrame(test_predictions.reshape(X_test.shape), columns=['latitude', 'longitude'])
test_predictions['type'] = 'model_prediction'
y_test_df = pd.DataFrame(y_test, columns=['latitude', 'longitude'])
y_test_df['type'] = 'y_test_true'
frames = [y_test_df, test_predictions]
pred_df = pd.concat(frames)

# losses = pd.DataFrame(history.history)
# losses.plot()
# plt.savefig('training.png', dpi = 600)

pred_plot = sns.scatterplot(x='latitude', y='longitude', hue='type', data=pred_df, alpha=0.6, linewidth=0, palette='viridis')
fig = pred_plot.get_figure()
fig.savefig('prediction_scatterplot.png', dpi = 600)
