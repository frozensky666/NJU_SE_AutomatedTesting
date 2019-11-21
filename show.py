import numpy
import matplotlib.pyplot as plt


def show(index):  # 显示从index开始的10张图片的前后对比
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    test_data = numpy.load("test_data/test_data.npy")
    attack_data = numpy.load("attack_data/attack_data.npy")
    test_labels = numpy.load("test_data/test_labels.npy")
    attack_pred_labels = numpy.load("attack_data/attack_pred_labels.npy")
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(test_data[i+index].reshape(28, 28))
        plt.xlabel(class_names[test_labels[i+index]])
        plt.subplot(2, 10, i+11)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(attack_data[i+index].reshape(28, 28))
        plt.xlabel(class_names[attack_pred_labels[i+index]])
    plt.show()


if __name__ == "__main__":
    show(9087)
