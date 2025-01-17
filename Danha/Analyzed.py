import pandas as pd
import tensorflow as tf
import numpy as np

# Load data from CSV, 문장 데이터&감정 라벨
data = pd.read_csv(r'C:\Users\bobos\OneDrive\Documents\GitHub\TensorFlow\Danha\Data.csv')

# Text data and labels
texts = data['text'].tolist()#리스트 변환
labels = data['sentiment'].values#numpy로 저장

# Manual tokenization - 문장을 단어로 쪼개고 정수 변환(매핑)&리스트에 저장장
word_index = {}
sequences = []
for text in texts:
    words = text.lower().split()# 소문자 변환& 내가 틀린 SPLITㅎㅎ
    sequence = []
    #단어가 리스트에 없으면 정수 매핑해서 추가하는 과정정
    for word in words:
        if word not in word_index:
            word_index[word] = len(word_index) + 1
        sequence.append(word_index[word])
    sequences.append(sequence)

# Padding sequences - 모델이 처리할 수 있게 문장 각각의 리스트 길이 맞춰주기기
max_length = max(len(sequence) for sequence in sequences)
padded_sequences = []
for sequence in sequences:
    padded_sequence = sequence[:max_length] + [0] * (max_length - len(sequence))
    padded_sequences.append(padded_sequence)

# Convert to numpy array
padded_sequences = np.array(padded_sequences)

#여기부터~

# Model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 16, input_length=max_length),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=15, verbose=1)

#~여기까지가 모델 만들고 학습시키는 과정

# Test the model
test_texts = ["The price was too high for the quality", "The interface is user-friendly", "I'm satisfied"]
#이거는 그냥 테스트용 문장 따로 만들어서 넣은 거 같은데 사용자 입력으로 쓰거나 데이터 수 늘려서 훈련/테스트 데이터 나눠서 쓰면 될 듯듯
#아래꺼는 테스트용 문장들 정수 변환하는 건데 위에랑 비슷한 거
test_sequences = []
for text in test_texts:
    words = text.lower().split()
    sequence = []
    for word in words:
        if word in word_index:
            sequence.append(word_index[word])
    test_sequences.append(sequence)

# Padding test sequences - 길이 맞추는 거, 이것도 위에랑 똑같다고 보면 됨
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

#여기부터는 모델 평가하는 과정정
# Evaluate the model
evaluation = model.evaluate(padded_sequences, labels, verbose=0)

# Extract loss and accuracy
loss = evaluation[0]
accuracy = evaluation[1]

# Print loss and accuracy
print("Loss:", loss)
print("Accuracy:", accuracy) 