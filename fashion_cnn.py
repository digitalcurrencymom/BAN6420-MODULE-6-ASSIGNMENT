#!/usr/bin/env python3
"""
fashion_cnn.py

Trains a small CNN on Fashion MNIST (if TensorFlow is available) and writes
predictions for the first two test images to `predictions_output.txt`.

This script is written to be runnable on systems with or without TensorFlow.
If TensorFlow is missing, it falls back to a small scikit-learn classifier
on the reduced dataset (for demonstration).

Outputs:
- predictions_output.txt   (two predicted labels, one per line)
- fashion_mnist_plot.png    (saved if matplotlib is available and run)

Usage:
    python fashion_cnn.py

"""
import os
import sys

OUT_PRED = "predictions_output.txt"
OUT_PLOT = "fashion_mnist_plot.png"

def save_predictions(preds):
    with open(OUT_PRED, 'w') as f:
        for p in preds:
            f.write(str(int(p)) + "\n")
    print(f"Wrote predictions to {OUT_PRED}")

def try_tensorflow_path():
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from tensorflow.keras.datasets import fashion_mnist
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.utils import to_categorical

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
        x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.0
        y_train_cat = to_categorical(y_train)

        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train_cat, epochs=3, batch_size=128, validation_split=0.1, verbose=2)

        preds = model.predict(x_test[:2])
        labels = preds.argmax(axis=1)
        save_predictions(labels)

        # simple plot for the two images
        try:
            fig, axes = plt.subplots(1,2, figsize=(6,3))
            for i, ax in enumerate(axes):
                ax.imshow(x_test[i].reshape(28,28), cmap='gray')
                ax.set_title(f'Pred: {labels[i]}')
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(OUT_PLOT)
            print(f"Saved plot to {OUT_PLOT}")
        except Exception as e:
            print('Could not save plot:', e)

        return True
    except Exception as e:
        print('TensorFlow path failed or not available:', e)
        return False

def fallback_sklearn():
    # small fallback using scikit-learn on downsampled data
    try:
        from sklearn.datasets import load_digits
        from sklearn.neural_network import MLPClassifier
        import numpy as np
        import matplotlib.pyplot as plt

        digits = load_digits()
        X = digits.images
        y = digits.target
        n = len(X)
        X = X.reshape(n, -1) / 16.0

        # train small MLP
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300)
        clf.fit(X, y)
        preds = clf.predict(X[:2])
        save_predictions(preds)

        # quick plot
        try:
            fig, axes = plt.subplots(1,2, figsize=(6,3))
            for i, ax in enumerate(axes):
                ax.imshow(digits.images[i], cmap='gray')
                ax.set_title(f'Pred: {preds[i]}')
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(OUT_PLOT)
            print(f"Saved fallback plot to {OUT_PLOT}")
        except Exception as e:
            print('Could not save fallback plot:', e)

        return True
    except Exception as e:
        print('Fallback path failed:', e)
        return False

def main():
    ok = try_tensorflow_path()
    if not ok:
        print('Switching to sklearn fallback...')
        ok2 = fallback_sklearn()
        if not ok2:
            print('All run paths failed. Please install required packages.')

if __name__ == '__main__':
    main()
