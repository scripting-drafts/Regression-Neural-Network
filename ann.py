import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skmultilearn.model_selection import iterative_train_test_split
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

X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.3)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(2,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2))
model.compile(optimizer='adam',loss='mse')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=20000, callbacks=[early_stop])

model.save('google_places_model.h5')

losses = pd.DataFrame(history.history)
losses.plot()
plt.savefig('training.png', dpi = 600)
