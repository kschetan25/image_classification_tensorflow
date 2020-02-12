import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

label_names = ["Dog", "Cat", "Car"]

# prepare image for detection by the trained model.


def process_image(file_path):
    IMG_SIZE = 70
    cv_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(cv_image, (IMG_SIZE, IMG_SIZE))
    return resized_image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# load the training model for detection
trained_model = tf.keras.models.load_model("image_classification_CNN.model")
image_to_predict = './test_images/cat.jpeg'
result = trained_model.predict([process_image(image_to_predict)])

# display the predicted image
cv_image = cv2.imread(
    image_to_predict, cv2.IMREAD_GRAYSCALE)
plt.imshow(cv_image, cmap='gray')
plt.show()


# Print the prediction
print(label_names[np.argmax(result)])
