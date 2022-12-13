import pretty_midi
import numpy as np
import pretty_midi
from music21 import *
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import glob
from itertools import groupby
import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import MultiLabelBinarizer

data_path = "./data/jazz_piano/" # modify here if you want to use other dataset      
encoded_data_path = "./data/encoded_jazz/"

def extract_midi_info(path):
    mid = converter.parse(path)
    max_idx = np.argmax(len(i) for i in mid)
    piano_part = mid[max_idx] #probably
    for i in piano_part:
        if isinstance(i, tempo.MetronomeMark):
            bpm = i.getQuarterBPM()
            break
    try:
        key = piano_part.keySignature
    except:
        print(f"Error while finding key signature for song {temp}")

    key_in_major = key.asKey(mode='major')
    offset_by = key_in_major.tonic.pitchClass
    return offset_by, bpm

def preprocess_midi(path, offset_by, bpm):
    mid = pretty_midi.PrettyMIDI(midi_file=path)
    filtered_inst_ls = [inst for inst in mid.instruments if ((len(inst.notes) > 0) and
                                                    (inst.is_drum == False) and
                                                    (inst.program < 8)
                                                   )]
    piano = filtered_inst_ls[np.argmax([len(inst.notes) for inst in filtered_inst_ls])]
            
    start_time = piano.notes[0].start
    end_time = piano.get_end_time()
    
    quater_note_len = 60/bpm
    nth_note = 8
    fs = 1/(quater_note_len/nth_note)
    
    piano_roll = piano.get_piano_roll(fs = fs, times = np.arange(start_time, end_time,1./fs))
    piano_roll = np.roll(piano_roll, -offset_by)
    out = np.where(piano_roll > 0, 1,0)
    
    return out.T

def process_piano_roll(piano_roll, max_consecutive = 64):
    prev = np.random.rand(128)
    count = 0
    remove_idxs = []
    remove_slice = []
    for idx, piano_slice in enumerate(piano_roll):
        if(np.array_equal(prev, piano_slice)):
            count+=1
            if (count > max_consecutive):
                remove_idxs.append(idx)
                if (str(piano_slice) not in remove_slice):
                    remove_slice.append(str(piano_slice))
        else:
            count = 0
        prev = piano_slice
    out_piano_roll = np.delete(piano_roll, remove_idxs, axis=0)
    return out_piano_roll

failed_list = []
# keep track of list of midi we failed to parse and preprocess
for temp in glob.glob(data_path + "*.mid"):
    try:
        print(temp)
        offset_by, bpm = extract_midi_info(temp)
        piano_roll = preprocess_midi(temp, offset_by, bpm)
        piano_roll = process_piano_roll(piano_roll)
        name  = temp.split("/")[-1].split(".")[0]
        out_name = encoded_data_path + f'encoded_{name}.npy'
        np.save(out_name, piano_roll)
        print(f"saved {out_name}")
        
    except:
        print(f"Faield to preprocess {temp}")
        failed_list.append(temp)
        continue


mlb = MultiLabelBinarizer()
mlb.fit([np.arange(128).tolist()])

# encoded_data_path = "./data/encoded_data/"
output_path = "./output/"

batch_size = 32
# sequence_length = 500
sequence_length = 600
generate_sample_every_ep = 100

maxlen = sequence_length  # Max sequence size
# embed_dim = 128  # Embedding size for each token
embed_dim = 128  # Embedding size for each token
num_heads = 4  # Number of attention heads
feed_forward_dim = 128  # Hidden layer size in feed forward network inside transformer

combi_to_int_pickle = "combi_to_int.pickle"
int_to_combi_pickle = "int_to_combi.pickle"
vocab_pickle = "vocab.pickle"

# vocab_size = 50000
vocab_size = 40000
unk_tag_str = '<UNK>'
unk_tag_idx = 0
pad_tag_str = ''
pad_tag_idx = 1


all_songs = []
all_songs_np = np.empty((0,128), np.int8)
for temp in glob.glob(encoded_data_path + "*.npy"):
    encoded_data = np.load(temp).astype(np.int8)
    all_songs.append(encoded_data)
    all_songs_np = np.append(all_songs_np, encoded_data, axis=0)

print(all_songs_np.shape)
unique_np, counts = np.unique(all_songs_np, axis=0, return_counts=True)

unique_note_intergerized = np.array(mlb.inverse_transform(unique_np))
count_sort_ind = np.argsort(-counts)

vocab = unique_note_intergerized[count_sort_ind][:vocab_size-2].tolist()
top_counts = counts[count_sort_ind][:vocab_size-1].tolist()

vocab.sort(key=len)
vocab.insert(unk_tag_idx, unk_tag_str)
vocab.insert(pad_tag_idx, pad_tag_str)
vocab_size = len(vocab)
print(f"vocab size: {len(vocab)}")
print(f"vocab first 5 words: {vocab[:5]}")

combi_to_int = dict((combi, number) for number, combi in enumerate(vocab))
int_to_combi = dict((number, combi) for number, combi in enumerate(vocab))

all_song_tokenised = []
for idx, song in enumerate(all_songs):
    print(f"processing song number {idx}")
    song = mlb.inverse_transform(song)
    song = [combi_to_int[tup] if tup in vocab else unk_tag_idx for tup in song]
#     song = [combi_to_int[tup] for tup in song]
    all_song_tokenised.append(np.array(song))
print(f"Completed tokenising all song")

#delete to free up memory
del all_songs
del all_songs_np
gc.collect()