from tensorflow.keras import layers, models, regularizers

CNN = models.Sequential([
    layers.Conv2D(16, (3, 3), padding='same', input_shape=(176, 208, 1)),
    layers.Conv2D(16, (3, 3), padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding='same'),
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(256, (3, 3), padding='same'),
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.7),

    layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(4),
    layers.Softmax()
])

