path = r"C:\Users\baekj\Documents\GitHub\TensorFlow\DataSets\sentiment.csv"

import pandas as pd
import tensorflow as tf
import numpy as np

# Load dataset
data = pd.read_csv(path)

texts = data['text'].tolist()
labels = data['sentiment'].values

# Tokenization and word indexing
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

# Padding sequences
max_length = max(len(sequence) for sequence in sequences)
padded_sequences = []
for sequence in sequences:
    padded_sequence = sequence[:max_length] + [0] * (max_length - len(sequence))
    padded_sequences.append(padded_sequence)

padded_sequences = np.array(padded_sequences)

# Positional encoding function
def get_positional_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, np.newaxis]
    div_terms = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_enc = np.zeros((max_len, d_model))
    pos_enc[:, 0::2] = np.sin(positions * div_terms)
    pos_enc[:, 1::2] = np.cos(positions * div_terms)
    return tf.constant(pos_enc, dtype=tf.float32)

# Define Transformer-based model
class TransformerSentiment(tf.keras.Model):
    def __init__(self, vocab_size, max_length, d_model, num_heads, ff_dim):
        super(TransformerSentiment, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, input_length=max_length)
        self.positional_encoding = get_positional_encoding(max_length, d_model)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x += self.positional_encoding
        attn_output = self.attention(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)
        x = self.global_pool(out2)
        return self.output_layer(x)

# Instantiate and compile the model
vocab_size = len(word_index) + 1
d_model = 16
num_heads = 2
ff_dim = 64
model = TransformerSentiment(vocab_size, max_length, d_model, num_heads, ff_dim)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=15, verbose=1)

# Interactive testing and adding new data
while True:
    # Input sentence
    text = input("Text: ")
    if text.lower() == "stop":
        break

    # Tokenize and pad input sentence
    words = text.lower().split()
    sequence = [word_index[word] for word in words if word in word_index]
    padded_sequence = sequence[:max_length] + [0] * (max_length - len(sequence))
    padded_sequence = np.array([padded_sequence])

    # Predict sentiment
    prediction = model.predict(padded_sequence)
    predicted_sentiment = np.argmax(prediction[0])
    print(f"Predicted Sentiment: {predicted_sentiment}")

    # Get actual sentiment for further training
    sentiment = input("Sentiment (0, 1, 2 or 'skip' to skip): ")
    if sentiment.lower() != "skip":
        try:
            sentiment = int(sentiment)
            if sentiment not in [0, 1, 2]:
                print("Invalid sentiment. Must be 0, 1, or 2.")
                continue

            # Add new data to training set
            labels = np.append(labels, sentiment)
            new_sequence = np.array([padded_sequence[0]])
            padded_sequences = np.vstack((padded_sequences, new_sequence))

            # Retrain the model
            model.fit(padded_sequences, labels, epochs=5, verbose=1)

        except ValueError:
            print("Invalid input. Please provide a valid integer or 'skip'.")

# Evaluate the model
evaluation = model.evaluate(padded_sequences, labels, verbose=0)
loss = evaluation[0]
accuracy = evaluation[1]
print("Loss:", loss)
print("Accuracy:", accuracy)
