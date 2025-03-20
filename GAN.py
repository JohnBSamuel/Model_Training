import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 64  # Image size (64x64)
LATENT_DIM = 100  # Noise vector size
NUM_CHANNELS = 3  # RGB images
BATCH_SIZE = 16  # Smaller batch for small dataset
EPOCHS = 200  # More epochs for better quality with small data
NUM_IMAGES_TO_GENERATE = 50  # Reduced to 50 synthetic images

# Load and preprocess images from ~/Desktop/DATASET
def load_images():
    image_paths = glob.glob(os.path.expanduser("~/Desktop/DATASET/*.jpg"))
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
            images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    images = np.array(images)
    print(f"Loaded {len(images)} images")
    return images

# Data augmentation to expand dataset
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build a deeper Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 512, input_dim=LATENT_DIM),
        layers.LeakyReLU(negative_slope=0.2),  # Updated from alpha
        layers.Reshape((8, 8, 512)),
        
        layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        
        layers.Conv2D(NUM_CHANNELS, (3, 3), padding='same', activation='tanh')
    ])
    return model

# Build a deeper Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

# Training step with explicit trainable toggle
@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        # Train discriminator
        discriminator.trainable = True
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Compute gradients
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Check gradients and apply
    if gen_gradients is None or any(g is None for g in gen_gradients):
        print("Warning: Generator gradients are None")
    if disc_gradients is None or any(g is None for g in disc_gradients):
        print("Warning: Discriminator gradients are None")
        return gen_loss, disc_loss  # Skip applying if None
    
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    return gen_loss, disc_loss

# Training loop
def train(dataset, generator, discriminator, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator)
        print(f"Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")
        
        # Generate sample images every 20 epochs
        if (epoch + 1) % 20 == 0:
            generate_and_save_images(generator, epoch + 1, test_noise=tf.random.normal([16, LATENT_DIM]))

# Generate and save 50 images
def generate_and_save_images(generator, epoch, test_noise=None, num_images=NUM_IMAGES_TO_GENERATE):
    save_dir = os.path.expanduser("~/Desktop/gan_output")
    os.makedirs(save_dir, exist_ok=True)
    
    if test_noise is None or test_noise.shape[0] < num_images:
        test_noise = tf.random.normal([num_images, LATENT_DIM])
    
    generated_images = generator(test_noise, training=False)
    generated_images = (generated_images + 1) * 127.5  # Rescale to [0, 255]
    
    for i in range(num_images):
        img = generated_images[i].numpy().astype("uint8")
        Image.fromarray(img).save(os.path.join(save_dir, f"generated_image_{i:03d}.jpg"))
    
    # Save a preview of up to 16 images
    preview_num = min(16, num_images)
    fig = plt.figure(figsize=(4, 4))
    for i in range(preview_num):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i].numpy().astype("uint8"))
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"preview_epoch_{epoch}.png"))
    plt.close()
    print(f"Generated and saved {num_images} images to {save_dir}")

# Main execution
def main():
    # Load and augment data
    images = load_images()
    if len(images) < 10:
        print("Too few images found in ~/Desktop/DATASET. Need at least 10 for training.")
        return
    
    # Augment images
    augmented_images = []
    for img in images:
        img = np.expand_dims(img, 0)  # Add batch dimension
        it = datagen.flow(img, batch_size=1)
        for _ in range(10):  # Generate 10 augmented versions per image
            augmented_images.append(next(it)[0])
    images = np.concatenate([images, augmented_images])
    print(f"After augmentation: {len(images)} images")
    
    dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(1000).batch(BATCH_SIZE)

    # Build models
    generator = build_generator()
    discriminator = build_discriminator()

    # Train the GAN
    train(dataset, generator, discriminator, EPOCHS)

    # Generate 50 images after training
    generate_and_save_images(generator, EPOCHS, num_images=NUM_IMAGES_TO_GENERATE)

if __name__ == "__main__":
    main()