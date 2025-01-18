import pandas as pd
import tensorflow as tf
import numpy as np

path = r"C:\Users\baekj\Documents\GitHub\TensorFlow\DataSets\sentiment.csv"

# load data from CSV
data = pd.read_csv(path)

# Text data and labels
texts = data['text'].tolist()
labels = data['sentiment'].values

# Manual tokenization - convert text to integers and store in a list
word_index = {}
sequences = []
for text in texts:
    words = text.lower().split()
    sequence = []
    for word in words:
        if word not in word_index:
            word_index[word] = len(word_index) + 1
        sequence.append(word_index[word])
    sequences.append(sequence)

# Padding sequences - pad sequences to the same length(max_length)
max_length = max(len(sequence) for sequence in sequences)
padded_sequences = []
for sequence in sequences:
    padded_sequence = sequence[:max_length] + [0] * (max_length - len(sequence))
    padded_sequences.append(padded_sequence)

padded_sequences = np.array(padded_sequences)

# Model architecture using Transformer
inputs = tf.keras.Input(shape=(max_length,))
embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1, 16)(inputs)
attention_output = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(embedding_layer, embedding_layer)
pooled_output = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
outputs = tf.keras.layers.Dense(3, activation='softmax')(pooled_output)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=15, verbose=1)

# Test the model
test_texts = ["The price was too high for the quality", "The interface is user-friendly", "I'm satisfied"]
test_sequences = []
for text in test_texts:
    words = text.lower().split()
    sequence = []
    for word in words:
        if word in word_index:
            sequence.append(word_index[word])
    test_sequences.append(sequence)

# Padding test sequences
padded_test_sequences = []
for sequence in test_sequences:
    padded_sequence = sequence[:max_length] + [0] * (max_length - len(sequence))
    padded_test_sequences.append(padded_sequence)

# Convert to numpy array
padded_test_sequences = np.array(padded_test_sequences)

# Make predictions
predictions = model.predict(padded_test_sequences)

# Print predicted sentiments
for i, text in enumerate(test_texts):
    print(f"Text: {text}, Predicted Sentiment: {np.argmax(predictions[i])}")

# Evaluate the model
evaluation = model.evaluate(padded_sequences, labels, verbose=0)

loss = evaluation[0]
accuracy = evaluation[1]

print("Loss:", loss)
print("Accuracy:", accuracy)