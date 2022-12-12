from model import *
import glob
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np

seq_len = 600
epochs = 1500
batchsize = 64
encoded_data_path = "./data/encoded_jazz/"

with open('./data/jazz_pickle/all_song_tokenised.pickle', 'rb') as f:
    processed_dataset = pickle.load(f)
with open('./data/jazz_pickle/vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
t_data = Generator(processed_dataset, batchsize, seq_len)
v_data = Generator(processed_dataset, batchsize, seq_len, val_split=0.1)
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
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile("adam", loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), None],)

model.summary()

start = processed_dataset[1][:seq_len-200]
output_path = f"./output/jazz_{epochs}{batchsize}{int(time.time())}/"

weight_path = output_path + "jazz.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weight_path,
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]

history = model.fit(x = t_data,
                    callbacks = callbacks_list,                    
                   epochs = epochs,
                   verbose = 1,
                   validation_data = v_data)

train_loss = []
val_loss = []
train_loss += history.history['loss']
val_loss += history.history['val_loss']

plt.plot(train_loss)
plt.plot(val_loss)
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'validation_loss'], loc='upper right')
plt.savefig(output_path + 'loss.png')
plt.show()