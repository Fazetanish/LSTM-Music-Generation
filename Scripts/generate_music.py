import pickle
import numpy as np
from keras.models import load_model
from music21 import instrument, note, stream, chord

def generate_music(model_path='music_generation_model.keras', pitchnames_path='pitchnames.pkl', sequence_length=100, num_notes=500, start_with=None):
    """Generate a musical sequence using the trained model"""
    
    # Load the model and pitchnames
    model = load_model(model_path)
    with open(pitchnames_path, 'rb') as f:
        pitchnames = pickle.load(f)
    
    # Create mapping dictionaries
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    # Either use provided starting sequence or generate a random one
    if start_with is None:
        start = np.random.randint(0, len(pitchnames)-sequence_length)
        pattern = [note_to_int[pitchnames[i]] for i in range(start, start+sequence_length)]
    else:
        pattern = start_with
    
    prediction_output = []
    
    # Generate notes
    for note_index in range(num_notes):
        # Prepare input for prediction
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(len(pitchnames))
        
        # Make prediction
        prediction = model.predict(prediction_input, verbose=0)
        
        # Get the index with highest probability
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        # Update pattern for next iteration (remove the first element, append the new prediction)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    
    return prediction_output

def create_midi(prediction_output, filename="generated_music.mid"):
    """Convert the output from the prediction to a MIDI file"""
    offset = 0
    output_notes = []
    
    # Create note and chord objects based on the values generated
    for pattern in prediction_output:
        # If the pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a single note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        
        # Increase offset to place things in time
        offset += 0.5  # You can adjust this to control note duration
    
    # Create a stream with the notes/chords
    midi_stream = stream.Stream(output_notes)
    
    # Write the MIDI file
    midi_stream.write('midi', fp=filename)
    print(f"Generated MIDI file saved as {filename}")

# Generate music and create a MIDI file
if __name__ == "__main__":
    # Generate sequence of notes
    prediction_output = generate_music(
        model_path='music_generation_model.keras', 
        pitchnames_path='pitchnames.pkl',
        num_notes=500  # Number of notes to generate
    )
    
    # Create MIDI file from the generated notes
    create_midi(prediction_output, "generated_music.mid")