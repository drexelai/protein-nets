from gan import GAN, discriminator, generator, latent_dim
import tensorflow as tf
from tensorflow import keras
import numpy as np
from data import get_data

def main(batch_size, file_dir):
    # Prepare the dataset. We use both the training & test MNIST digits.
    x = get_data(file_dir)

    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    # To limit the execution time, we only train on 100 batches. You can train on
    # the entire dataset. You will need about 20 epochs to get nice results.
    gan.fit(dataset.take(100), epochs=1)
if __name__ == '__main__':
    main()