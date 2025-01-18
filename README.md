## Introduction

Fake audio can be used maliciously to spread misinformation. This project leverages Deep Learning(RNN(LSTM)) to create a model capable of detecting fake audio, providing a robust tool to combat audio-based misinformation.

This project proposes an automated solution to classify audio as real or fake using Recurrent Neural Networks (LSTM).
It preprocesses raw audio files using Librosa, extracts features, and trains a robust model using TensorFlow.
Streamlit is used to create an interactive web-based interface for real-time testing.

### Dataset
#### Hierarchy :
for-2sec
  ├── training
  │     ├── fake
  │     └── real
  ├── validation
  │     ├── fake
  │     └── real
  └── testing
        ├── fake
        └── real

Dataset Link : https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

for-2sec(Dataset):Based on the for-norm folder, but truncated at 2 seconds.
