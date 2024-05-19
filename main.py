import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist #done to import the dataset directly from tensorflow and no need to download csv files

#Split into training data and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()  #This function helps in splitting directly into 2 tuples which contain the training and test data for the processing of the model.

#Normalizing (Scaling down)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#creating the model
model = tf.keras.models.Sequential()
#Add layers
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) #Output layer (10 for 10 digits)

#Model compiling
model.compile(optimizer='adam', loss ='sparse_categorical_crossentropy', metrics=['accuracy'])
#Model Fitting
model.fit(x_train, y_train, epochs=3)

model.save('handwrittenmodel.h5')

model = tf.keras.models.load_model('handwrittenmodel.h5')

#Model Evaluation
loss, accuracy =  model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

#Dataset analyzing
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}") #Argmax gives us the index of the field that has the highest number. eg which neuron has the highest activation
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number+=1