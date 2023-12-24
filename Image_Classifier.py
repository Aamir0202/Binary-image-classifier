import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import shutil
from shutil import copyfile
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

source_path = 'kagglecatsanddogs/PetImages'

source_path_dogs = os.path.join(source_path, 'Dog')
source_path_cats = os.path.join(source_path, 'Cat')

# Deletes all non-image files (there are two .db files bundled into the dataset)
# !find /tmp/PetImages/ -type f ! -name "*.jpg" -exec rm {} +

# os.listdir returns a list containing all files under the given path
print(f"There are {len(os.listdir(source_path_dogs))} images of dogs.")
print(f"There are {len(os.listdir(source_path_cats))} images of cats.")

# Define root directory
root_dir = 'kagglecatsanddogs/cats-v-dogs'

# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
  shutil.rmtree(root_dir)

# GRADED FUNCTION: create_train_val_dirs
def create_train_val_dirs(root_path):

  train_dir = os.path.join(root_path, 'training')
  validation_dir = os.path.join(root_path, 'validation')

  os.makedirs(train_dir)
  os.makedirs(validation_dir)

  # Directory with training cat/dog pictures
  train_cats_dir = os.path.join(train_dir, 'cats')
  train_dogs_dir = os.path.join(train_dir, 'dogs')

  os.makedirs(train_cats_dir)
  os.makedirs(train_dogs_dir)

  # Directory with validation cat/dog pictures
  validation_cats_dir = os.path.join(validation_dir, 'cats')
  validation_dogs_dir = os.path.join(validation_dir, 'dogs')

  os.makedirs(validation_cats_dir)
  os.makedirs(validation_dogs_dir)

try:
  create_train_val_dirs(root_path=root_dir)
except FileExistsError:
  print("File already exists!!")

for rootdir, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        print(os.path.join(rootdir, subdir))
        
def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):

  files = []
  for filename in os.listdir(SOURCE_DIR):
      file_name = os.path.join(SOURCE_DIR, filename)
      if os.path.getsize(file_name) > 0:
          files.append(filename)
      else:
          print("{} has zero length so discarding".format(filename))
  train_set_size = int(len(files)*SPLIT_SIZE)
  shuffled_data = random.sample(files, len(files))
  training_data = shuffled_data[0:train_set_size]
  testing_data = shuffled_data[train_set_size:len(files)]

  for file in training_data:
      src_file = os.path.join(SOURCE_DIR, file)
      des_file = os.path.join(TRAINING_DIR, file)
      copyfile(src_file, des_file)

  for file in testing_data:
      src_file = os.path.join(SOURCE_DIR, file)
      des_file = os.path.join(VALIDATION_DIR, file)
      copyfile(src_file, des_file)

CAT_SOURCE_DIR = "kagglecatsanddogs/PetImages/Cat/"
DOG_SOURCE_DIR = "kagglecatsanddogs/PetImages/Dog/"

TRAINING_DIR = "kagglecatsanddogs/cats-v-dogs/training/"
VALIDATION_DIR = "kagglecatsanddogs/cats-v-dogs/validation/"

TRAINING_CATS_DIR = os.path.join(TRAINING_DIR, "cats/")
VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, "cats/")

TRAINING_DOGS_DIR = os.path.join(TRAINING_DIR, "dogs/")
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, "dogs/")

# Empty directories in case you run this cell multiple times
if len(os.listdir(TRAINING_CATS_DIR)) > 0:
  for file in os.scandir(TRAINING_CATS_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_DOGS_DIR)) > 0:
  for file in os.scandir(TRAINING_DOGS_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_CATS_DIR)) > 0:
  for file in os.scandir(VALIDATION_CATS_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_DOGS_DIR)) > 0:
  for file in os.scandir(VALIDATION_DOGS_DIR):
    os.remove(file.path)

# Define proportion of images used for training
split_size = .9

# NOTE: Messages about zero length images should be printed out
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)

# Your function should perform copies rather than moving images so original directories should contain unchanged images
print(f"\n\nOriginal cat's directory has {len(os.listdir(CAT_SOURCE_DIR))} images")
print(f"Original dog's directory has {len(os.listdir(DOG_SOURCE_DIR))} images\n")

# Training and validation splits
print(f"There are {len(os.listdir(TRAINING_CATS_DIR))} images of cats for training")
print(f"There are {len(os.listdir(TRAINING_DOGS_DIR))} images of dogs for training")
print(f"There are {len(os.listdir(VALIDATION_CATS_DIR))} images of cats for validation")
print(f"There are {len(os.listdir(VALIDATION_DOGS_DIR))} images of dogs for validation")

# Check if a saved model exists, and load it if available
saved_model_path = 'cat_VS_dog_classifier.h5'

if os.path.exists(saved_model_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model(saved_model_path)
    print("Model loaded successfully.")
    
else:
    def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
        """
        Creates the training and validation data generators

        Args:
            TRAINING_DIR (string): directory path containing the training images
            VALIDATION_DIR (string): directory path containing the testing/validation images

        Returns:
            train_generator, validation_generator - tuple containing the generators
        """
        
        train_datagen = ImageDataGenerator(rescale= 1./255)

        # Pass in the appropriate arguments to the flow_from_directory method
        train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                            batch_size=20,
                                                            class_mode='binary',
                                                            target_size=(150, 150))

        # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
        validation_datagen = ImageDataGenerator(rescale= 1./255)

        # Pass in the appropriate arguments to the flow_from_directory method
        validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                        batch_size=20,
                                                                        class_mode='binary',
                                                                        target_size=(150, 150))
        
        return train_generator, validation_generator

    train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

    def create_model():

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=RMSprop(learning_rate=1e-4),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        return model

    # Get the untrained model
    model = create_model()

    # Train the model
    history = model.fit(train_generator,
                        epochs=15,
                        verbose=1,
                        validation_data=validation_generator)
    
    # Save the trained model for future use
    model.save(saved_model_path)
    print("Model trained and saved successfully.")

# Function to load and classify the selected image
def classify_image():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                           filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
    
    if file_path:
        img = Image.open(file_path)
        img = img.resize((150, 150))  # Resize the image to match the model input size
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.vstack([img])

        # Predict the class
        prediction = model.predict(img, batch_size=1)[0]

        result_text.set("Prediction: " + ("Dog" if prediction > 0.5 else "Cat"))

        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Create the main application window
app = tk.Tk()
app.title("Cat vs Dog Classifier")

# Create a button to choose an image
choose_button = tk.Button(app, text="Choose Image", command=classify_image)
choose_button.pack(pady=10)

# Create a label to display the prediction result
result_text = tk.StringVar()
result_label = tk.Label(app, textvariable=result_text, font=("Helvetica", 16))
result_label.pack(pady=10)

# Create a label to display the selected image
image_label = tk.Label(app)
image_label.pack(pady=10)

# Run the application
app.mainloop()
