from generate import generate
import numpy
if __name__ == "__main__":
    count = 10000  # 选择前几张测试图片
    shape = (count, 28, 28, 1)  # generate 输入输出格式
    model_shape = (count, 28, 28)  # model 输入格式

    # 从npy文件中加载数据
    test_images = numpy.load("test_data/test_data.npy")

    images_after = generate(
        test_images[:count].reshape(shape), shape)  # 生成对抗样本
    numpy.save("./attack_data/attack_data.npy", images_after)  # 存储对抗样本
