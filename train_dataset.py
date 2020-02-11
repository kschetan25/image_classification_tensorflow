#Training the model for prediction in future.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

# read the x_train pickle data
read_x_train = open("x_train.pickle", "rb")
x_train = pickle.load(read_x_train)

# read the x_train pickle data
read_y_train = open("y_train.pickle", "rb")
y_train = pickle.load(read_y_train)

#Normalizing the data
x_train = x_train/255.0

# Building the CNN network to train the model
dense_layers = [0]
layer_sizes = [64]
convolution_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for convolution_layer in convolution_layers:
            graph_name = "image_classification_grapg_{}".format(int(time.time()))
            print(graph_name)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for i in range(convolution_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for j in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(3))
            model.add(Activation('softmax'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(graph_name))

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(x_train, y_train,
                      batch_size=32,                            #send n images in a batch
                      epochs=10,                                #train the model for n epochs
                      validation_split=0.3,                     #split train and test data
                      callbacks=[tensorboard])

#Save the model
model.save('image_classification_CNN.model')

#print the accuracy on validation data
val_loss, val_acc = model.evaluate(x_train, y_train)
print(val_loss, val_acc)