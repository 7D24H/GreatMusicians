import rnn
import numpy as np
import tensorflow as tf
from get_data import get_data
from get_data import notes_to_midi
from get_data import play_midi

# reuse model to make predictions
def reuse(note_count, output_path='generated.mid', model_path='model.h5'):
    model = tf.keras.models.load_model(model_path)
    notes = [x+21 for x in generate(model, note_count)]
    notes_to_midi(notes, output_path, "Acoustic Grand Piano")


def generate(model, note_count):
    seed = tf.random.uniform(shape=[1, 40, 1],minval=0,maxval=1, dtype=tf.float32) ## TODO: make 40 a hyperparameter, hardcoding for now

    generated_notes = []
    for i in range(note_count):
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0  # diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        # index_N = index / float(L_symb)
        generated_notes.append(index)
        norm_index = index / 88
        seed = np.insert(seed[0], len(seed[0]), norm_index)
        seed = seed[1:]
        seed = np.reshape(seed, (1, 40, 1))
    return generated_notes


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_data()

    ## train and save the model
    ## TODO: uncomment the following line to train the model
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # model = rnn.get_model(y_train.shape[1])
    # history = model.fit(X_train, y_train, batch_size=100, epochs=5, validation_data=(X_test, y_test))
    # model.save("model.h5")
    # print("Model saved to model.h5")

    ## reuse the local model to make predictions
    reuse(100)
    play_midi('generated.mid')
