import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from model import CNN

train_dir = '../datasets/train/'
validation_dir = '../datasets/test/'
epochs = 20

train_datagen = ImageDataGenerator(rescale=1./255,
    #rotation_range=10,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #zoom_range=0.2,
    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255,
    #rotation_range=10,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #zoom_range=0.2,
    fill_mode='nearest')
'''
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
'''

train_dataset = train_datagen.flow_from_directory(
        train_dir,
        target_size=(176, 208),
        batch_size=20,
        class_mode='sparse',
        color_mode='grayscale')

test_dataset = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(176, 208),
        batch_size=20,
        class_mode='sparse',
        color_mode='grayscale')

try:
  CNN = load_model('model.keras')
  print('Loaded existing model.')
except:
  print('Generated new model.')

optimizer = Adam(learning_rate=0.00001)
CNN.compile(optimizer=optimizer,
            # maybe make true idk
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1, 
    patience=5,
    min_lr=10**-8,
    verbose=1
)

checkpoint_cb = ModelCheckpoint(
    'backups/backup_e{epoch:02d}.keras',
    period=10,
    save_weights_only=False
)

#CNN.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
CNN.fit(train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[lr_scheduler, checkpoint_cb])
test_loss, test_acc = CNN.evaluate(test_dataset, verbose=2)
print('\nTest loss:{test_loss}\nTest accuracy:{test_acc}', test_acc)
CNN.save('backup.keras')
