import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Воспроизводимость
np.random.seed(42)
tf.random.set_seed(42)

def plot_history(history, title):
    """Строит графики loss и accuracy для train и val."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='val', linewidth=2)
    axes[0].set_title(f'{title} — Loss')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['accuracy'], label='train', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='val', linewidth=2)
    axes[1].set_title(f'{title} — Accuracy')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def compute_metrics(model, x_test, y_test_true, name):
    """
    Считает метрики: accuracy, precision, recall, F1 (macro и weighted).
    Возвращает словарь с метриками и массив предсказаний.
    """
    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    result = {
        'name':        name,
        'accuracy':    accuracy_score(y_test_true, y_pred),
        'precision':   precision_score(y_test_true, y_pred,
                                       average='macro', zero_division=0),
        'recall':      recall_score(y_test_true, y_pred,
                                    average='macro', zero_division=0),
        'f1_macro':    f1_score(y_test_true, y_pred,
                                average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test_true, y_pred,
                                average='weighted', zero_division=0),
        'y_pred':      y_pred
    }

    print(f"\n=== {name} ===")
    print(f"Accuracy:         {result['accuracy']:.4f}")
    print(f"Precision macro:  {result['precision']:.4f}")
    print(f"Recall macro:     {result['recall']:.4f}")
    print(f"F1 macro:         {result['f1_macro']:.4f}")
    print(f"F1 weighted:      {result['f1_weighted']:.4f}")

    return result


def show_classification_report(y_test_true, y_pred, name):
    """Печатает детальный отчёт по каждому классу."""
    print(f"\n--- Classification report: {name} ---")
    print(classification_report(y_test_true, y_pred, digits=4))


def show_confusion_matrix(y_test_true, y_pred, title, cmap='Blues'):
    """Рисует confusion matrix."""
    cm = confusion_matrix(y_test_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                cbar=False, square=True)
    plt.title(title)
    plt.xlabel('Предсказано')
    plt.ylabel('Истина')
    plt.tight_layout()
    plt.show()
    

(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()


# Нормализация: пиксели были 0..255 → станут 0..1
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# One-hot encoding меток
y_train_full_cat = to_categorical(y_train_full, 10)
y_test_cat = to_categorical(y_test, 10)

# ВАЖНО: специально берём маленькую выборку,
# чтобы вызвать переобучение у базовой модели

x_train = x_train_full[:3000]
y_train = y_train_full_cat[:3000]

x_val = x_train_full[3000:6000]
y_val = y_train_full_cat[3000:6000]

print(f"Train: {x_train.shape}")
print(f"Val:   {x_val.shape}")
print(f"Test:  {x_test.shape}")
print(f"Классов: {y_train_full_cat.shape[1]}")


model_base = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dropout(0.4),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),   # скрытый слой
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),   # скрытый слой
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),   # скрытый слой
    
    layers.Dense(10, activation='softmax'),  # выход: 10 классов
])

model_base.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history_base = model_base.fit(
    x_train, y_train, 
    epochs=100, 
    batch_size=128,
    callbacks=[early_stop],
    validation_data=(x_val, y_val)
)

# Графики обучения
plot_history(history_base, "Регуляризованная модель")


result_base = compute_metrics(model_base, x_test, y_test, "Регуляризованная модель")


show_classification_report(y_test, result_base['y_pred'], "Регуляризованная модель")


# Confusion matrix
# show_confusion_matrix(
    # y_test, result_base['y_pred'],
    # title='Confusion Matrix — Базовая модель',
    # cmap='Reds'
# )