# почти доделал (розетка зоряди меня енегрией, я так болбше нэ можу)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Воспроизводимость
np.random.seed(42)
tf.random.set_seed(42)

# Настройки графиков
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 11

dataset = load_dataset("stanfordnlp/imdb")
print(dataset)

train_texts  = dataset['train']['text']
train_labels = dataset['train']['label']
test_texts   = dataset['test']['text']
test_labels  = dataset['test']['label']

y_train = np.array(train_labels)
y_test = np.array(test_labels)

print(f"Обучающих отзывов: {len(train_texts)}")
print(f"Тестовых отзывов:   {len(test_texts)}")
print(f"Форма y_train: {y_train.shape}, y_test: {y_test.shape}")
print(f"Распределение меток в train (0, 1): {np.bincount(y_train)}")


vocab_size = 10000
max_len    = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="")
tokenizer.fit_on_texts(train_texts)

print(f"Всего уникальных слов в train: {len(tokenizer.word_index)}")
print(f"Оставляем только {vocab_size} самых частых")

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

X_train = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
X_test  = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')


print(f"Форма X_train: {X_train.shape}")
print(f"Форма X_test:  {X_test.shape}")


model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, mask_zero=True),       # (batch, 50, 16)
    SpatialDropout1D(0.2),
    GlobalAveragePooling1D(),                           # (batch, 16)
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    Dense(1, activation='sigmoid')                        # (batch, 1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

print("Обучение завершено!")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='train loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='val loss',   linewidth=2)
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("Binary Crossentropy")
axes[0].set_title("Функция потерь (train vs val)")
axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='train accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='val accuracy',   linewidth=2)
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Точность (train vs val)")
axes[1].legend()

plt.tight_layout()
plt.show()