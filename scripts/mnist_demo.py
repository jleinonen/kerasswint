import numpy as np
from tensorflow import keras

from kerasswint import swint


def create_model(num_classes=10, input_shape=(32,32,1)):
    ip = keras.Input(shape=input_shape)
    x = swint.PatchEmbedding2D(64)(ip)
    x = swint.DualSwinTransformerBlock2D(64, dropout=0.1, num_heads=8)(x)
    x = swint.PatchMerge2D(128)(x)
    x = swint.SwinTransformer2D(128, dropout=0.1, num_heads=16)(x)
    x = swint.SwinTransformer2D(128, dropout=0.1, num_heads=16)(x)
    x = keras.layers.Activation("relu")(x)
    x = swint.PatchMerge2D(128, size=4)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=ip, outputs=x)


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    def preprocess(x):
        x = x.astype(np.float32) / 255
        xp = np.zeros((x.shape[0], 32, 32, 1), dtype=np.float32)
        xp[:,2:-2,2:-2,0] = x
        return xp

    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    return (x_train, y_train), (x_test, y_test)

def train_model(model, x_train, y_train, batch_size=128, epochs=1):
    model.compile(loss="sparse_categorical_crossentropy",
        optimizer="adam", metrics=["accuracy"])
    
    model.fit(x_train, y_train, batch_size=batch_size,
        epochs=epochs, validation_split=0.1)
