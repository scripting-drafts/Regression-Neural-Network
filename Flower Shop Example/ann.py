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

df = pd.read_csv('/home/gerard/2020-10-28_00:59:21_barcelona-floristeria.csv', delimiter=';')
df = df.drop('address', axis=1).drop('type', axis=1).drop('errors', axis=1)

keywords = ['flor', 'Flor']

df_filter = df.name.isin(keywords)
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

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2))
model.compile(optimizer='adam',loss='mse')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=12000, callbacks=[early_stop])

test_predictions = model.predict(X_test)
test_predictions = pd.DataFrame(test_predictions.reshape(X_test.shape), columns=['model_predict_lat', 'model_predict_lon'])
pred_df = pd.DataFrame(y_test, columns=['y_test_lat', 'y_test_lon'])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
print(pred_df)

# model.save('google_places_model.h5')
# later_model = load_model('google_places_model.h5')

losses = pd.DataFrame(history.history)
losses.plot()
plt.savefig("training.png")
