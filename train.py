import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from setting.general import TRAIN_SET_PATH, TEST_SET_PATH, IMAGE_RESHAPING_FACTOR, KERNEL_PATH
from infrastructure.model import build_model


EPOCHS = 10


def prepare_record(record):
    return np.array(record[1:]).reshape((-1, IMAGE_RESHAPING_FACTOR, IMAGE_RESHAPING_FACTOR, 1))


def prepare_output(record):
    output = np.zeros(10)
    output[record[0]] = 1
    return output.reshape((1, 10))


class_names = list(range(10))

train_data = pd.read_csv(TRAIN_SET_PATH)
train_data = train_data.apply(lambda x: pd.Series([prepare_record(x / 255), prepare_output(x)], index=['input', 'output']), axis=1)

test_data = pd.read_csv(TEST_SET_PATH)
test_data = test_data.apply(lambda x: pd.Series([prepare_record(x / 255), prepare_output(x)], index=['input', 'output']), axis=1)

model = build_model()
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_dataset = tf.data.Dataset.from_tensor_slices((train_data['input'].values.tolist(), train_data['output'].values.tolist()))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data['input'].values.tolist(), test_data['output'].values.tolist()))

history = model.fit(train_dataset, epochs=EPOCHS,
                    validation_data=test_dataset)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

model.save(KERNEL_PATH)
print('Saved trained model at %s ' % KERNEL_PATH)

scores = model.evaluate(test_dataset, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
