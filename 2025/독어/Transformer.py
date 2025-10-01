import tensorflow as tf
import numpy as np

# 1. 데이터 로드
# 귀찮아서 그냥 다 여기 옮겨적을거임.
de_sentences = ["Hallo Welt", "Ich liebe dich", "Wie geht es dir?"]
en_sentences = ["Hello world", "I love you", "How are you?"]
ko_sentences = ["안녕 세상", "사랑해", "잘 지내?"]

# 예시: 독-한 / 독-영 / 영-한 병렬
de_ko_pairs = list(zip(de_sentences, ko_sentences))
de_en_pairs = list(zip(de_sentences, en_sentences))
en_ko_pairs = list(zip(en_sentences, ko_sentences))

# 2. 토크나이저 준비
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def build_tokenizer(sentences, num_words=1000):
    tokenizer = Tokenizer(num_words=num_words, filters='')
    tokenizer.fit_on_texts(sentences)
    return tokenizer

# 각각의 언어 토크나이저
de_tokenizer = build_tokenizer(de_sentences)
en_tokenizer = build_tokenizer(en_sentences)
ko_tokenizer = build_tokenizer(ko_sentences)

def encode_pairs(pairs, src_tok, tgt_tok, maxlen=10):
    src = src_tok.texts_to_sequences([s for s, _ in pairs])
    tgt = tgt_tok.texts_to_sequences([t for _, t in pairs])
    src = pad_sequences(src, maxlen=maxlen, padding='post')
    tgt = pad_sequences(tgt, maxlen=maxlen, padding='post')
    return src, tgt

de_ko_src, de_ko_tgt = encode_pairs(de_ko_pairs, de_tokenizer, ko_tokenizer)
de_en_src, de_en_tgt = encode_pairs(de_en_pairs, de_tokenizer, en_tokenizer)
en_ko_src, en_ko_tgt = encode_pairs(en_ko_pairs, en_tokenizer, ko_tokenizer)

# 3. Transformer 모델
from tensorflow.keras import layers

def build_transformer(src_vocab, tgt_vocab, embed_dim=64, num_heads=2, ff_dim=128):
    inputs = layers.Input(shape=(None,))
    x = layers.Embedding(src_vocab, embed_dim)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dense(embed_dim)(x)
    outputs = layers.Dense(tgt_vocab, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

de_ko_model = build_transformer(len(de_tokenizer.word_index)+1, len(ko_tokenizer.word_index)+1)
de_en_model = build_transformer(len(de_tokenizer.word_index)+1, len(en_tokenizer.word_index)+1)
en_ko_model = build_transformer(len(en_tokenizer.word_index)+1, len(ko_tokenizer.word_index)+1)

# 4. 학습
de_ko_model.fit(de_ko_src, np.expand_dims(de_ko_tgt, -1), epochs=1000, verbose=0)
de_en_model.fit(de_en_src, np.expand_dims(de_en_tgt, -1), epochs=1000, verbose=0)
en_ko_model.fit(en_ko_src, np.expand_dims(en_ko_tgt, -1), epochs=1000, verbose=0)

# 5. 번역 함수
def translate(sentence, model, src_tok, tgt_tok, maxlen=10):
    seq = src_tok.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(seq, verbose=0)
    pred_ids = np.argmax(pred[0], axis=-1)
    inv_map = {v:k for k,v in tgt_tok.word_index.items()}
    return " ".join([inv_map.get(i, '') for i in pred_ids if i > 0])

# 6. 직접 번역 vs 간접 번역 비교
test_sentence = "Ich liebe dich"

direct = translate(test_sentence, de_ko_model, de_tokenizer, ko_tokenizer)
pivot_en = translate(test_sentence, de_en_model, de_tokenizer, en_tokenizer)
indirect = translate(pivot_en, en_ko_model, en_tokenizer, ko_tokenizer)

print("독→한 직접 번역:", direct)
print("독→영 중간 번역:", pivot_en)
print("독→영→한 간접 번역:", indirect)