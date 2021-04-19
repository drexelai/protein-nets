from gan import GAN, discriminator, generator, latent_dim
import tensorflow as tf
from tensorflow import keras
import numpy as np
from data import get_data
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from loadptn import x_min, y_min, z_min, x_max, y_max, z_max, atom_pos, atom_type

def main(epochs, batch_size, file_dir, train=True):
    # Prepare the dataset. We use both the training & test MNIST digits.
    not_padded = get_data(file_dir)
    # Change [37,26,32,38] to [40,27,32,38]
    x = np.pad(not_padded, ((0, 0),(1, 2), (1, 0), (0, 0), (0, 0)), 'mean')
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim, batch_size=batch_size)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
    )
    # To limit the execution time, we only train on 100 batches. You can train on
    # the entire dataset. You will need about 20 epochs to get nice results.
    if train:
        history = gan.fit(x, batch_size=batch_size, epochs=epochs)
        gan.discriminator.save('gan_disciminator')
        gan.generator.save('gan_generator')

        g_loss, d_loss = history.history['g_loss'], history.history['d_loss']
        plt.plot(g_loss)
        plt.plot(d_loss)
        plt.xticks(np.arange(0, epochs, step=1))  # Set label locations.
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Protein Structure Generation With DCGAN')
        plt.legend(['Generator loss', 'Discriminator loss'], loc='upper right')
        plt.show()
    else:
        gan.discriminator = keras.models.load_model('gan_disciminator')
        gan.generator = keras.models.load_model('gan_generator')
def get_accuracies(pred, labels, threshold=.5):
    pred_output = pred.copy()
    labels_output = labels.copy()

    pred_output[pred_output >= threshold] = 1
    pred_output[pred_output < threshold] = 0

    labels_output[labels_output >= threshold] = 1
    labels_output[labels_output < threshold] = 0

    accuracies = []
    for i in range(pred_output.shape[0]):
        accuracies.append(accuracy_score(labels_output[i], pred_output[i]))
    return accuracies
    #print(classification_report(labels_output,pred_output))


# Plot Accuracy and Loss
def plot_training_loss(history):
	# Plot training & validation accuracy values
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()


if __name__ == '__main__':
    epochs = 50
    batch_size = 16
    main(epochs, batch_size, 'ptn11H_1000')
