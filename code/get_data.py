import os

import numpy as np
import tensorflow as tf
import pretty_midi
from sklearn.model_selection import train_test_split

filepath = "../data/debussy/"  ## only songs by Debussy are used.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000

import pygame


def play_midi(midi_file):
    """
    Play midi files
    """
    def play_music(music_file):
        """
        stream music with mixer.music module in blocking manner
        this will stream the sound from disk while playing
        """
        clock = pygame.time.Clock()
        try:
            pygame.mixer.music.load(music_file)
        except pygame.error:
            return
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            # check if playback has finished
            clock.tick(30)

    freq = 44100  # audio CD quality
    bitsize = -16  # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 1024  # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)

    try:
        # use the midi file you just saved
        play_music(midi_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit

def get_notes_names(notes):
    return np.vectorize(pretty_midi.note_number_to_name)(notes)

def parse_midi_to_notes(midi_file):
    """
    Parse midi files
    """
    pm = pretty_midi.PrettyMIDI(midi_file)

    instrument = pm.instruments[0]

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

    ## more specific information about the notes
    #
    # instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    # print('Instrument name:', instrument_name)

    # prev_start = sorted_notes[0].start
    # for note in instrument.notes:
    #     start = note.start
    #     end = note.end
    #     print("pitch", note.pitch)
    #     print("start", start)
    #     print("end", end)
    #     print("step", start - prev_start)
    #     print("duration", end - start)
    #     prev_start = start

    notes = [note.pitch for note in sorted_notes]
    return notes

def get_data():
    """
    Classical Music MIDI https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi
    """
    midi_file_names = []
    for filename in os.listdir(filepath):
        if filename.endswith(".mid"):
            midi_file_names.append(filepath + filename)

    features, labels = [], []
    for file in midi_file_names:
        notes = parse_midi_to_notes(file)
        length = 40 ## TODO: hyperparameter, (maybe related to how many notes in a bar)
        for i in range(len(notes) - length):
            features.append(notes[i:i+length])
            labels.append(notes[i+length])

    ## https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
    ## pitch range for piano: 21 - 108 (piano has 88 keys), we map 21 to 0, 108 to 87
    X = (np.reshape(features, (len(features), length, 1)) - 21) / 1.0 ## expand one dimension for LSTM
    # use one-hot encoding for labels
    y = (np.array(labels)-21)
    y = tf.keras.utils.to_categorical(y, 88)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def notes_to_midi(
    notes,
    out_file: str,
    instrument_name: str,
    velocity: int = 100,  # note loudness
    step: float = 0.25,  # time between notes ## maybe randomness?
    duration: float = 0.25,  # note duration ## maybe randomness?
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for note in notes:
    start = float(prev_start + step)
    end = float(start + duration)
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm

if __name__ == "__main__":
    # try playing midi file
    midi_file = "../data/debussy/DEB_CLAI.MID"
    play_midi(midi_file)

