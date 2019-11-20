import numpy
import tensorflow as tf
from tensorflow import keras


def generate(images: numpy.ndarray, shape: tuple):
    # 生成对抗样本
    generate_images: numpy.ndarray = images

    return generate_images.reshape(shape)


if __name__ == "__main__":

    count = 4399

    test_images = numpy.load("test_data/test_data.npy")
    test_labels = numpy.load("test_data/test_labels.npy")
    model = keras.models.load_model("model.h5")
    shape = (count, 28, 28, 1)  # generate 输入输出格式
    model_shape = (count, 28, 28)  # model 输入格式
    # model.summary()
    loss, acc = model.evaluate(
        generate(test_images[:count], shape).reshape(model_shape), test_labels[:count])
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))   # 88.77%
