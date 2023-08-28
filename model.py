import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# CNN Model
Classifier = Sequential()

Classifier.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 3)))
Classifier.add(BatchNormalization())
Classifier.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
Classifier.add(BatchNormalization())
Classifier.add(MaxPooling2D(pool_size=(2, 2)))

Classifier.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
Classifier.add(BatchNormalization())
Classifier.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
Classifier.add(BatchNormalization())
Classifier.add(MaxPooling2D(pool_size=(2, 2)))

Classifier.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
Classifier.add(BatchNormalization())
Classifier.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
Classifier.add(BatchNormalization())
Classifier.add(MaxPooling2D(pool_size=(2, 2)))

Classifier.add(Flatten())

Classifier.add(Dropout(0.25))
Classifier.add(Dense(1024, activation='relu'))
Classifier.add(BatchNormalization())
Classifier.add(Dense(512, activation='relu'))
Classifier.add(BatchNormalization())

Classifier.add(Dense(units=50, activation='softmax'))
Classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Classifier.summary()

# plot_model(Classifier, 'Classifier.png', show_shapes=True)

# Data generator
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, rotation_range=25)
validate_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

balanced_training_set = train_datagen.flow_from_directory('Dataset/backup_train', target_size=(28, 28),
                                                 batch_size=32, class_mode='categorical', shuffle=True)
validate_set = validate_datagen.flow_from_directory('Dataset/backup_val', target_size=(28, 28), batch_size=30,
                                                    class_mode='categorical', shuffle=True)

test_set = test_datagen.flow_from_directory('Dataset/backup_test', target_size=(28, 28), batch_size=30,
                                            class_mode='categorical', shuffle=True)

# Train model
history = Classifier.fit(balanced_training_set, steps_per_epoch=375, epochs=10, validation_data=validate_set,
                         validation_steps=40)
# Evaluate the model on the test set
evaluation = Classifier.evaluate(test_set, steps=len(test_set))
accuracy = evaluation[1] * 100

print(f"Test accuracy: {accuracy:.2f}%")

accuracy = int(accuracy)

model_save_path = f'./accuracy_{accuracy}.h5'
Classifier.save(model_save_path)

print(f"Model saved at {model_save_path}")


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()