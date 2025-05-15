import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from preprocess import get_notes_from_midis, prepare_sequences
import os
import pickle

# Try to load preprocessed data
if os.path.exists('data.pkl'):
    print("Loading preprocessed data...")
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
        network_input = data['network_input']
        network_output = data['network_output']
        pitchnames = data['pitchnames']
        n_vocab = data['n_vocab']
else:
    # Extract and preprocess
    print("Extracting notes from MIDI files...")
    notes = get_notes_from_midis('./Midi Files/**/*.mid')

    print("Preparing sequences...")
    sequence_length = 100
    network_input, network_output, n_vocab, pitchnames = prepare_sequences(notes, sequence_length)

    # Save preprocessed data
    with open('data.pkl', 'wb') as f:
        pickle.dump({
            'network_input': network_input,
            'network_output': network_output,
            'pitchnames': pitchnames,
            'n_vocab': n_vocab
        }, f)


# Extract notes from MIDI files
print("Extracting notes from MIDI files...")
notes = get_notes_from_midis('./Midi Files/**/*.mid')

# Prepare sequences for LSTM
print("Preparing sequences...")
sequence_length = 100  # Adjust based on your needs
network_input, network_output, n_vocab, pitchnames = prepare_sequences(notes, sequence_length)

# Build the LSTM model
print("Building model...")
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(n_vocab))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Set up checkpoint to save weights during training
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.keras"
checkpoint = ModelCheckpoint(
    filepath, 
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]

# Train the model
print("Training model...")
model.fit(
    network_input, 
    network_output,
    epochs=50,  # You may need more epochs for good results
    batch_size=64,
    callbacks=callbacks_list
)

# Save the model
model.save('music_generation_model.keras')
# Save the pitchnames for later use
import pickle
with open('pitchnames.pkl', 'wb') as f:
    pickle.dump(pitchnames, f)