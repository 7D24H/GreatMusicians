from model import *
import glob
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit([np.arange(128).tolist()])

train_loss = []
val_loss = []
sequence_length = 600
generate_sample_every_ep = 100
maxlen = sequence_length
embed_dim = 128
num_heads = 4
feed_forward_dim = 128
vocab_size = 40000
unk_tag_str = '<UNK>'
unk_tag_idx = 0
pad_tag_str = ''
pad_tag_idx = 1
encoded_data_path = ""

all_songs = []
all_songs_np = np.empty((0,128), np.int8)
for temp in glob.glob(encoded_data_path + "*.npy"):
    encoded_data = np.load(temp).astype(np.int8)
    all_songs.append(encoded_data)
    all_songs_np = np.append(all_songs_np, encoded_data, axis=0)

unique_np, counts = np.unique(all_songs_np, axis=0, return_counts=True)

unique_note_intergerized = np.array(mlb.inverse_transform(unique_np))
count_sort_ind = np.argsort(-counts)

vocab = unique_note_intergerized[count_sort_ind][:vocab_size-2].tolist()
top_counts = counts[count_sort_ind][:vocab_size-1].tolist()

vocab.sort(key=len)
vocab.insert(unk_tag_idx, unk_tag_str)
vocab.insert(pad_tag_idx, pad_tag_str)
vocab_size = len(vocab)

with open('./16v2/combi_to_int.pickle', 'rb') as f:
    combi_to_int = pickle.load(f)
    
with open('./16v2/all_song_tokenised.pickle', 'rb') as f:
    all_song_tokenised = pickle.load(f)

with open('./16v2/int_to_combi.pickle', 'rb') as f:
    int_to_combi = pickle.load(f)
    
with open('./16v2/vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)


inputs = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block1 = TransformerBlock(embed_dim, num_heads, feed_forward_dim, dropout_rate = 0.25)
transformer_block2 = TransformerBlock(embed_dim, num_heads, feed_forward_dim, dropout_rate = 0.25)
transformer_block3 = TransformerBlock(embed_dim, num_heads, feed_forward_dim, dropout_rate = 0.25)
transformer_block4 = TransformerBlock(embed_dim, num_heads, feed_forward_dim, dropout_rate = 0.25)
transformer_block5 = TransformerBlock(embed_dim, num_heads, feed_forward_dim, dropout_rate = 0.25)
x = transformer_block1(x)
x = transformer_block2(x)
x = transformer_block3(x)
x = transformer_block4(x)
x = transformer_block5(x)
outputs = tf.keras.layers.Dense(vocab_size)(x)
model = tf.keras.Model(inputs=inputs, outputs=[outputs, x])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile("adam", loss=[loss_fn, None],)

start_tokens = all_song_tokenised[1][:sequence_length-200]
num_tokens_generated = 80
gen_callback = GeneratorCallback(num_tokens_generated, start_tokens, print_every= generate_sample_every_ep)

epochs = 1500
batchsize = 64
output_path = f"./output/pop_{epochs}{batchsize}{int(time.time())}_16v2f/"


my_training_batch_generator = Generator(all_song_tokenised, batchsize, sequence_length)
my_validation_batch_generator = Generator(all_song_tokenised, batchsize, sequence_length, val_split=0.1)

weight_path = output_path + "music-gen-weight.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weight_path,
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint,gen_callback]

history = model.fit(x = my_training_batch_generator,
                    callbacks = callbacks_list,                    
                   epochs = epochs,
                   verbose = 1,
                   validation_data = my_validation_batch_generator)

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
print("Result stored in {}".format(output_path))