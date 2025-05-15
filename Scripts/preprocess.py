import glob
import os
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.utils import to_categorical

def get_notes_from_midis(midi_path):
    """ Extract all notes and chords from multiple midi files """

    # Dynamically resolve the MIDI directory path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MIDI_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Midi Files'))
    midi_path = os.path.join(MIDI_DIR, '**', '*.mid')
    
    notes = []
    
    for file in glob.glob(midi_path, recursive=True):
        try:
            midi = converter.parse(file)
            print(f"Parsing {file}")
            
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
                
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            
    return notes

def prepare_sequences(notes, sequence_length=100):
    """ Prepare the sequences for the LSTM """
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    network_input = []
    network_output = []
    
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    
    n_patterns = len(network_input)
    n_vocab = len(pitchnames)
    
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)
    
    return network_input, network_output, n_vocab, pitchnames
