from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from loadptn import x_min, y_min, z_min, x_max, y_max, z_max, atom_pos, atom_type
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass
from sklearn.metrics import classification_report, confusion_matrix
print((z_max-z_min, y_max-y_min, x_max-x_min, 1 + len(atom_type) + len(atom_pos)))
# Create the discriminator
discriminator = keras.Sequential(
    [
        keras.Input(shape=(z_max-z_min, y_max-y_min, x_max-x_min, 1 + len(atom_type) + len(atom_pos))),
        layers.Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling3D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator
latent_dim = 1 + len(atom_type) + len(atom_pos)
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(5*3*8* latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((5, 3, 8, latent_dim)),
        layers.Conv3DTranspose(latent_dim, (4, 4, 4), strides=(2, 3, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv3DTranspose(latent_dim, (4, 4, 4), strides=(4, 3, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv3D(latent_dim, (7, 7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)



class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, batch_size):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions_d = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions_d)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}
    def call(inputs):
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        return self.generator(random_latent_vectors)

