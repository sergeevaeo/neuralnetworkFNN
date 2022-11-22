import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

sns.despine()

from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Nadam
from keras.layers import LeakyReLU
from keras import regularizers


# перетасовывание двух массивов в унисон
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# подготовление данных для обучения, создание массива
# X_train, Y_train - тренировочные данные,
# X_test- Y_test- тестовые данные,
# для проверки точности того, как данные соответствуют модели
def create_Xt_Yt(X, Y, percentage=0.9):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = Y[0:p]

    X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[p:]
    Y_test = Y[p:]

    return X_train, X_test, Y_train, Y_test


# график цен на акции с 2005 года по 2017
data = pd.read_csv('AAPL.csv')
#data = data.loc[:, 'Adj Close'].tolist()
data = data.loc[:, 'Adj Close'].pct_change().dropna().tolist()
plt.plot(data)
plt.show()

# кол-во дней, на которых нейросеть будет учиться
WINDOW = 30

# кол-во шагов вперед, на которые нейросеть должна предсказать
FORECAST = 5

# Простой способ создания временных окон
# для обучения нашей нейронной сети мы
# получим следующие пары X, Y: цены в
# момент закрытия рынка за 30 дней и [1, 0] или [0, 1]
# в зависимости от того, выросло или упало значение цены
# для бинарной классификации
X, Y = [], []
for i in range(0, len(data), 1):
    try:
        x_i = data[i:i + WINDOW]
        y_i = data[i + WINDOW + FORECAST]

        last_close = x_i[WINDOW - 1]
        next_close = y_i

        #if last_close < next_close:
            #y_i = [1, 0]
        #else:
            #y_i = [0, 1]

    except Exception as e:
        print(e)
        break

    X.append(x_i)
    Y.append(y_i)

# Будем нормализовать наши 30-дневные окна с помощью z-score
# - это мера относительного разброса наблюдаемого или измеренного значения
# Для задачи регрессии так уже сделать не получится,
# ведь если мы будем также вычитать среднее и делить
# на отклонение, нам придется восстанавливать это значение
# для значения цены в следующий день, а там уже эти параметры
# могут быть совершенно другими
#X = [(np.array(x) - np.mean(x)) / np.std(x) for x in X]
X, Y = np.array(X), np.array(Y)

X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)

'''
#
#
# тут начало создания модели, 2 слоя
model = Sequential()
#  В процессе регуляризации мы накладываем определенные ограничения на веса нейронной сети
model.add(Dense(100, input_dim=30,
                activity_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU())
# это случайное игнорирование некоторых весов, чтобы нейроны не выучивали одинаковые признаки
model.add(Dropout(0.5))
model.add(Dense(100,
                activity_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(2))
model.add(Activation('softmax'))
'''

model = Sequential()
model.add(Dense(100, input_dim=30,
                activity_regularizer=regularizers.l2(0.00001)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(100,
                activity_regularizer=regularizers.l2(0.00001)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(1))
model.add(Activation('linear'))

#
#
#
# шаг градиентного спуска
opt = Nadam(learning_rate=0.001)

#
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
model.compile(optimizer=opt,
              loss='mse',
              metrics=['accuracy'])

#
# Обучение нейронной сети
history = model.fit(X_train, Y_train,
                    epochs=150,
                    batch_size=128,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks=[reduce_lr, checkpointer],
                    shuffle=True)

# Процесс обучения завершен, можно вывести графики на экран
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='best')
plt.show()

from sklearn.metrics import confusion_matrix

#
pred = model.predict(np.array(X_test))

C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])

print(C / C.astype(np.float64).sum(axis=1))

FROM = 0
TO = FROM + 500

original = Y_test[FROM:TO]
predicted = pred[FROM:TO]

plt.plot(predicted, color='blue', label='Predicted data')
plt.plot(original, color='black', label='Original data')
plt.legend(loc='best')
plt.title('Actual and predicted from point %d to point %d of test set' % (FROM, TO))
plt.show()
