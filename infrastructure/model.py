from numpy import array
from tensorflow.keras import layers, models
from setting.general import IMAGE_RESHAPING_FACTOR, KERNEL_PATH


def normalize(pic_array):
    vector = array(pic_array).reshape((-1, IMAGE_RESHAPING_FACTOR, IMAGE_RESHAPING_FACTOR, 1))
    return vector / 255


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(IMAGE_RESHAPING_FACTOR, (3, 3),
                            activation='softmax',
                            input_shape=(IMAGE_RESHAPING_FACTOR, IMAGE_RESHAPING_FACTOR, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='softmax'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='softmax'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='softmax'))
    model.add(layers.Dense(10))

    return model


class Classification:
    __class_names__ = list(range(10))

    def __init__(self):
        self.model = build_model()
        self.model.load_weights(KERNEL_PATH)

    def predict(self, pic_array):
        result_vector = self.model.predict(normalize(pic_array))[0]
        class_predictions = dict(zip(self.__class_names__, result_vector))
        most_valuable = {k: v for k, v in sorted(class_predictions.items(),
                                                 key=lambda item: item[1],
                                                 reverse=True)}
        return list(most_valuable.keys())[0]
