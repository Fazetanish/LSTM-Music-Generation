# 🎵 LSTM Music Generator

This project is a music generation model built using an LSTM (Long Short-Term Memory) neural network. It is trained on a dataset of MIDI files to learn musical patterns and generate new sequences in the style of the training data.

## 🚀 Features

- Preprocessing of MIDI files using `music21`
- Sequence generation using a 3-layer LSTM neural network
- Model checkpoints for best performance
- Music generation from a trained model

## 🧠 Model Architecture

- 3 stacked LSTM layers with 256 units each
- Dropout & BatchNormalization for regularization
- Dense output with `softmax` activation for note prediction
- Categorical cross-entropy loss with RMSprop optimizer

This model was trained on the classical music midi dataset on kaggle(https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi?resource=download).
