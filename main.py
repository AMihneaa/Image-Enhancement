import cv2
import tensorflow as tf
from keras import layers, models, Input
import os
import keras
import numpy as np
keras.config.disable_traceback_filtering()

checkpoint_path = "training_1/cp.keras"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=1,mode='min')

def psnr(image_true, image_pred):
    assert image_true.shape == image_pred.shape
    assert image_true.dtype == image_true.dtype
    mse = np.mean((image_true - image_pred) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(image_true)
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return psnr

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Pyhsical GPUs,",len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    def denoising_model(input_shape):
        # Encoder
        inputs = Input(shape=input_shape)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)

        # Decoder
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling2D()(x)
        outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

        model = models.Model(inputs, outputs)
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)
        return model


    # Define your dataset and preprocessing steps here
X_train = os.path.join('../train_noisy')
Y_train = os.path.join('../train')
X_val = os.path.join('../val_noisy')
Y_val = os.path.join('../val')

def preprocess_images(image_paths, target_size=(256, 256)):
    num_images = len(image_paths)
    preprocessed_images = np.zeros((num_images, *target_size, 3), dtype=np.float32)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)  # Load the image
        image = cv2.resize(image, target_size)  # Resize the image
        preprocessed_images[i] = image


    # Normalize pixel values to range [0, 1]
    preprocessed_images /= 255.0

    return preprocessed_images

    # Create and compile the model
image_trainx = [os.path.join(X_train, filename) for filename in os.listdir(X_train)]
image_trainy = [os.path.join(Y_train, filename) for filename in os.listdir(Y_train)]
image_valx = [os.path.join(X_val, filename) for filename in os.listdir(X_val)]
image_valy = [os.path.join(Y_val, filename) for filename in os.listdir(Y_val)]

X_train_preprocessed = preprocess_images(image_trainx)
Y_train_preprocessed = preprocess_images(image_trainy)
X_val_preprocessed = preprocess_images(image_valx)
Y_val_preprocessed = preprocess_images(image_valy)

height, width, channels = X_train_preprocessed[0].shape
model = denoising_model(input_shape=(height, width, channels))
model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
model.fit(X_train_preprocessed, Y_train_preprocessed, validation_data=(X_val_preprocessed, Y_val_preprocessed), epochs=20, batch_size=32, callbacks=[cp_callback])

    # Evaluate the model
loss = model.evaluate(X_val_preprocessed, image_valy)
print("Test Loss:", loss)
for i in range(len(X_val_preprocessed)):
    predicted_image = model.predict(X_val_preprocessed[i:i+1])
    psnr_value = psnr(Y_val_preprocessed[i], predicted_image[0])
    print("PSNR for image {}: {}".format(i+1,psnr_value))