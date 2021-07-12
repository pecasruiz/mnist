from tensorflow import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Pasamos la imagen de prayscale a binaria
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Flatten images
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilamos el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos
model.fit(x_train, y_train, epochs=10)

# Comprobamos la precision
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)
