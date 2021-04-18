from gan import GAN, discriminator, generator, latent_dim
import tensorflow as tf
from tensorflow import keras
import numpy as np
from data import get_data
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main(batch_size, file_dir):
    # Prepare the dataset. We use both the training & test MNIST digits.
    x = get_data(file_dir)
    
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
    )
    # To limit the execution time, we only train on 100 batches. You can train on
    # the entire dataset. You will need about 20 epochs to get nice results.
    print(generator.summary())
    print(discriminator.summary())
    history = gan.fit(x, batch_size=batch_size, epochs=20)
    g_loss, d_loss = history.history['g_loss'], history.history['d_loss']
    plt.plot(g_loss)
    plt.plot(d_loss)
    plt.xticks(np.arange(0, 20, step=1))  # Set label locations.
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Protein Structure Generation With DCGAN')
    # print(xticks(np.arange(0, 20, step=1)))
    # pred = np.stack(history.history['pred'], axis=0)
    # labels = np.stack(history.history['label'], axis=0)
    # accuracies = get_accuracies(pred, labels)
    # plt.plot(accuracies)
    plt.legend(['Generator loss', 'Discriminator loss'], loc='upper right')
    plt.show()
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
    main(10, 'ptn11H_10')
