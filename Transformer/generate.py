from model import *
import tensorflow as tf
import numpy as np
import random
import pretty_midi
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit([np.arange(128).tolist()])


with open('./data/jazz_pickle/all_song_tokenised.pickle', 'rb') as f:
    processed_dataset = pickle.load(f)
with open('./data/jazz_pickle/int_to_combi.pickle', 'rb') as f:
    int_to_combi = pickle.load(f)
with open('./data/jazz_pickle/vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)

seq_len = 600
vocab_size = len(vocab)
inputs = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32)
embedding_layer = TokenAndPositionEmbedding(seq_len, vocab_size, 128)
x = embedding_layer(inputs)
x = TransformerBlock(128, 4, 128)(x)
x = TransformerBlock(128, 4, 128)(x)
x = TransformerBlock(128, 4, 128)(x)
x = TransformerBlock(128, 4, 128)(x)
x = TransformerBlock(128, 4, 128)(x)
outputs = tf.keras.layers.Dense(vocab_size)(x)
model = tf.keras.Model(inputs=inputs, outputs=[outputs, x])

model.summary()
model.load_weights("./output/jazz.hdf5")

song_idx = random.randint(0,len(processed_dataset)-1)
start_idx = random.randint(0,50)   
sequence = processed_dataset[song_idx][start_idx:start_idx + 100].tolist()
while (sequence == [()]*seq_len):
    print("Got all zeros, rerolling")
    song_idx = random.randint(0,len(processed_dataset)-1)
    start_tokens = processed_dataset[song_idx][start_idx:start_idx + seq_len].tolist()
    
seq_copy = sequence.copy()

def sample_from(logits):
    logits, indices = tf.math.top_k(logits, k= 10, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = np.asarray(logits).astype("float32")
    # unk token
    if(0 in indices):
        unk_tag_position = np.where(indices == 0)[0].item()
        indices = np.delete(indices, unk_tag_position)
        preds = np.delete(preds, unk_tag_position)
    temp = np.exp(preds - np.max(preds))
    softmax_preds = temp / temp.sum(axis=0)
    return np.random.choice(indices, p=softmax_preds)

def convertToRoll(seq_list):
    seq_list = [int_to_combi[i] for i in seq_list]
    roll = mlb.transform(seq_list)
    print(seq_list)
    return roll

i = 0

while i <= 1000:

    x = sequence[-seq_len:]
    idx = -1
    if seq_len - len(sequence) > 0:
        x = sequence + [0] * (seq_len - len(sequence))
        idx = len(sequence) - 1    
    x = np.array([x])
    y, _ = model.predict(x)
    sequence.append(sample_from(y[0][idx]))
    i += 1
    print(f"generated {i} notes")
    

piano_roll = convertToRoll(sequence)
print("-------------------------------------------")
seq_copy = convertToRoll(seq_copy)

def piano_to_midi(piano_roll_in, fs, program=0, velocity = 64):
    piano_roll = np.where(piano_roll_in == 1, 64, 0)
    notes, _ = piano_roll.shape
    pm = pretty_midi.PrettyMIDI(initial_tempo=100.0)
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')
    print(piano_roll.shape)
    prev = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*np.nonzero(np.diff(piano_roll).T)):
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if prev[note] == 0:
            note_on_time[note] = time
            prev[note] = velocity
    pm.instruments.append(instrument)
    return pm

fs = 1/((60/150)/4)
name = "jazz"
output_path = "./output/"
mid_out = piano_to_midi(piano_roll.T, fs=fs)
mid_seq_copy = piano_to_midi(seq_copy.T, fs=fs)
## save the generated music
mid_out.write(output_path+f"generated-{name}.mid")
## save the 4 bars of trigger
mid_seq_copy.write(output_path+f"ori-snapshot.mid")

## save the full song of the trigger
# song_full = processed_dataset[song_idx][start_idx:].tolist()
# song_full = convertToRoll(song_full)
# song_full = piano_to_midi(song_full.T, fs=fs)
# midi_song_full_path = output_path+f"ori-full-{name}.mid"
# song_full.write(midi_song_full_path)