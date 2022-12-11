from model import *
from train import *
import pretty_midi

# model.load_weights("./output/classic_100641670557849_16v2f/music-gen-weight.hdf5")

song_idx = random.randint(0,len(all_song_tokenised)-1)
seq_start_at = random.randint(0,len(all_song_tokenised[song_idx])-sequence_length)   
start_tokens = all_song_tokenised[song_idx][seq_start_at:seq_start_at + 100].tolist()
while (start_tokens == [()]*sequence_length):
    print("Got all zeros, rerolling")
    song_idx = random.randint(0,len(all_song_tokenised)-1)
    seq_start_at = random.randint(0,len(all_song_tokenised[song_idx])-sequence_length)   
    start_tokens = all_song_tokenised[song_idx][seq_start_at:seq_start_at + sequence_length].tolist()
    
ori = start_tokens.copy()
backup = ori.copy()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sample_from(logits):
    logits, indices = tf.math.top_k(logits, k= 10, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = np.asarray(logits).astype("float32")
    if(unk_tag_idx in indices):
        unk_tag_position = np.where(indices == unk_tag_idx)[0].item()
        indices = np.delete(indices, unk_tag_position)
        preds = np.delete(preds, unk_tag_position)
    preds = softmax(preds)
    return np.random.choice(indices, p=preds)

def convertToRoll(seq_list):
    seq_list = [int_to_combi[i] for i in seq_list]
    roll = mlb.transform(seq_list)
    print(seq_list)
    return roll


tokens_generated = []
num_tokens_generated = 0
num_note_to_gen = 1000

while num_tokens_generated <= num_note_to_gen:

    x = start_tokens[-sequence_length:]
    pad_len = maxlen - len(start_tokens)
    sample_index = -1
    if pad_len > 0:
        x = start_tokens + [0] * pad_len
        sample_index = len(start_tokens) - 1
    
    x = np.array([x])
    y, _ = model.predict(x)
    sample_token = sample_from(y[0][sample_index])
    tokens_generated.append(sample_token)
    start_tokens.append(sample_token)
    num_tokens_generated = len(tokens_generated)
    print(f"generated {num_tokens_generated} notes")
    
# print(f"Piano int seq generated")
piano_roll = convertToRoll(start_tokens)
print("-------------------------------------------")
ori = convertToRoll(ori)

def piano_roll_to_pretty_midi(piano_roll_in, fs, program=0, velocity = 64):
    piano_roll = np.where(piano_roll_in == 1, 64, 0)
    notes, _ = piano_roll.shape
    pm = pretty_midi.PrettyMIDI(initial_tempo=100.0)
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')
    print(piano_roll.shape)
    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

bpm = 150
fs = 1/((60/bpm)/4)
name = "test"
mid_out = piano_roll_to_pretty_midi(piano_roll.T, fs=fs)
mid_ori = piano_roll_to_pretty_midi(ori.T, fs=fs)
midi_out_path = output_path+f"gpt-v3-id-{name}.mid"
if midi_out_path is not None:
        mid_out.write(midi_out_path)
        
midi_ori_path = output_path+f"ori-gpt-v3-id-{name}.mid"
if midi_ori_path is not None:
        mid_ori.write(midi_ori_path)

ori_full = all_song_tokenised[song_idx][seq_start_at:].tolist()
ori_full = convertToRoll(ori_full)
ori_full = piano_roll_to_pretty_midi(ori_full.T, fs=fs)
midi_ori_full_path = output_path+f"orifull-gpt-v3-{name}.mid"
if midi_ori_full_path is not None:
        ori_full.write(midi_ori_full_path)