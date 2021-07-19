from tensorflow import keras
from matplotlib import pyplot as plt


# Cargamos los datos
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Pasamos la imagen de prayscale a binaria
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Create the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation='softmax')
])

# Compilamos el modelo
# optimizer = keras.optimizers.Adam(lr=0.003)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos
history = model.fit(x_train, y_train, epochs=30,
                    batch_size=100, validation_split=0.2)


# PLot del acurracy y las perdidas
fig, (ax1, ax2) = plt.subplots(1, 2)

# summarize history for accuracy
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set(xlabel='epoch', ylabel='accuracy')
ax1.legend(['train', 'test'], loc='upper left')

# summarize history for loss
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set(xlabel='epoch', ylabel='loss')
ax2.legend(['train', 'test'], loc='upper left')
plt.show()


# Comprobamos la precision (set de test)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)
