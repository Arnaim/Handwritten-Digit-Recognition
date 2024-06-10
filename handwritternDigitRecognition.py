import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


mnist = tf.keras.datasets.mnist

# x = image, y = classification
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

"""
#creating model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#training model
model.fit(x_train, y_train, epochs=3)
model.save('handwrittern.model')

#commenting after training and saving model

"""

model = tf.keras.models.load_model('handwrittern.model')


#evaluating model accuracy
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)


numbers = 0
while os.path.isfile(f"numbers/num{numbers}.png"):
    try:
        img = cv2.imread(f"numbers/num{numbers}.png")[:,:,0]
        img = cv2.resize(img, (28, 28))  # Resize image to 28x28 if it's not already
        img = np.invert(img)  # Invert colors
        img = img / 255.0  # Normalize image
        img = img.reshape(1, 28, 28)  # Reshape image for prediction
        img = tf.keras.utils.normalize(img, axis=1)  # Normalize input as done during training
       
        prediction = model.predict(img)
        print(f"This digit is probably: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        numbers += 1
