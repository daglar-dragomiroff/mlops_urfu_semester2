import pickle
import tensorflow as tf

with open('cats_dogs.pkl', 'rb') as file:
    xception_network_cats_and_dogs = pickle.load(file)

IMAGE_SIZE = (180, 180)
img = tf.keras.preprocessing.image.load_img(
    "dataset_cats_dogs/PetImages/Dog/0.jpg", target_size=IMAGE_SIZE
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = xception_network_cats_and_dogs.predict(img_array)
score = predictions[0]
print(f"Score: {score}")
print(
    "С вероятностью %.2f процентов на картинке изображена кошка, с вероятностью %.2f процентов - собака."
    % (100 * (1 - score), 100 * score)
)