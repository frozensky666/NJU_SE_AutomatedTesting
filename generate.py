import numpy
import tensorflow as tf
from tensorflow import keras
from random import randint
import copy
import time
import sim
import os

# import gc
# import tracemalloc
# import matplotlib.pyplot as plt

# 忽略警告：
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 强行忽略警告：
# WARNING:tensorflow: load_from_saved_model (from tensorflow.python.keras.saving.saved_model_experimental) is deprecated and will be removed in a future version.
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ok = 0  # 存储算法中的“攻击成功”次数


def generate(images: numpy.ndarray, shape: tuple):

    # 生成对抗样本
    if numpy.max(images[0]) > 1:
        images = images/255.0
    global ok
    start_time = time.time()  # 开始时间
    count = shape[0]  # 图片数量
    model_shape = (count, 28, 28)  # model 输入格式
    model = tf.keras.experimental.load_from_saved_model("model")  # 载入模型
    res = numpy.array([])

    for i in range(0, shape[0]):
        pred_label = numpy.argmax(model.predict(
            numpy.reshape(images[i], (1, 28, 28))))
        img_after = attack(model, images[i].reshape(
            28*28), pred_label, 5, 0.3)
        res = numpy.concatenate((res, img_after.reshape(28*28)))
        print("\rNo.{}      Success: {}{}{}      Success_rate: {:.2f}      Runtime: {:.2f}s".format(
            i+1, ok, "/", count, ok/(i+1), time.time()-start_time), end='      ')
        tmp_sim = sim.SIM(images[i].reshape(
            28, 28), img_after.reshape(28, 28))
        print("sim: {:.2f}".format(tmp_sim), end="")
    print()
    ok = 0
    return res.reshape(shape)


def randomInt():  # 随机产生像素点攻击位置
    return randint(0, 28*28-1)


def attack_neighbors(image_d1, locale, attack_range, offset):  # 按照十字星的方式攻击locale附近的像素
    # image_d1: 图片一维载入
    # locale: 攻击位置
    # attack_range: 攻击范围(十字的臂长)
    # offset: 像素偏移量
    row = locale//28
    col = locale % 28
    image_d1[locale] += offset
    for i in range(1, attack_range):
        if(col+i < 28):
            image_d1[row*28+col+i] += offset
        if(col-i >= 0):
            image_d1[row*28+col-i] += offset
        if(row+i < 28):
            image_d1[(row+i)*28+col] += offset
        if(row-i >= 0):
            image_d1[(row-i)*28+col] += offset


def flags_on(flag: list, locale, attack_range):
    # 用来标记被攻击过的位置
    # list: 存储标记
    # locale: 攻击位置
    # attack_range: 攻击范围
    row = locale//28
    col = locale % 28
    flag[locale] == 3
    for i in range(1, attack_range):
        if(col+i < 28):
            flag[row*28+col+i] = 3
        if(col-i >= 0):
            flag[row*28+col-i] = 3
        if(row+i < 28):
            flag[(row+i)*28+col] = 3
        if(row-i >= 0):
            flag[(row-i)*28+col] = 3


def attack(model, image_d1, label, num_iters, offset=0.3):
    # model :模型
    # image :单张图片(一维 28*28, 归一化 )
    # label :图片实际分类
    # num_iters :成功攻击次数
    # offset: 单像素最大偏移量

    # starttime = time.time()
    pred_edge = 0.3  # 预测率降低到该值以下表示我认为的攻击成功
    image = copy.deepcopy(image_d1)
    flag = [0 for i in range(0, 28*28)]
    pred1 = model.predict(numpy.reshape(image, (1, 28, 28)))[0][label]
    # print("pred_before:", pred1)
    j = 0  # 攻击成功次数
    count = 0  # 攻击次数统计(random失败也会少量增加该值，以防死循环)
    attack_range = 28  # 十字星臂长
    while j <= num_iters and count <= 100 and pred1 > 0.3:
        # 攻击次数大于num_iter/ 攻击次数上限100 / 预测率小于等于0.3来作为攻击成功标志(实际上攻击可能没有成功，但是成功的概率很大)
        tmp = randomInt()
        if flag[tmp] == 3:
            count += 0.1
            continue
        count += 1
        attack_neighbors(image.reshape(28*28), tmp,
                         attack_range, offset)  # 正向攻击(像素点加偏移)
        pred2 = model.predict(numpy.reshape(image, (1, 28, 28)))[0][label]
        if pred2 < pred1:
            pred1 = pred2
            j += 1
            flag[tmp] = 1  # 正向攻击成功标记
            continue
        attack_neighbors(image.reshape(28*28), tmp,
                         attack_range, 0-2*offset)  # 反向攻击(像素点减去偏移)
        pred2 = model.predict(numpy.reshape(image, (1, 28, 28)))[0][label]
        if pred2 < pred1:
            pred1 = pred2
            j += 1
            flag[tmp] = 2
            continue
        attack_neighbors(image.reshape(28*28), tmp,
                         attack_range, offset)  # 攻击失败，恢复图像
        flags_on(flag, tmp, attack_range)  # 攻击无效标记,以后不再攻击相关像素点
    if pred1 <= 0.3:
        global ok
        ok += 1

    # print("pred_after: {}".format(pred1))
    # endtime = time.time()
    # print("攻击时间：", endtime-starttime)
    return image.reshape(1, 28, 28)


if __name__ == "__main__":

    # tracemalloc.start()  # 开始跟踪内存分配

    count = 100  # 选择前几张测试图片
    shape = (count, 28, 28, 1)  # generate 输入输出格式
    model_shape = (count, 28, 28)  # model 输入格式

    # 从npy文件中加载数据
    test_images = numpy.load("test_data/test_data.npy")
    # attack_images = numpy.load("attack_data/attack_data.npy")
    # test_labels = numpy.load("test_data/test_labels.npy")

    # model = keras.models.load_model("model.h5") #正常加载模型，但是这种加载出来的模型在循环调用中会内存泄漏
    # tf.keras.experimental.export_saved_model(model, "model") #神秘的存储方式，存在model文件夹中,这玩意儿跑起来贼快

    images_after = generate(
        test_images[:count].reshape(shape), shape)  # 生成对抗样本
    # numpy.save("./attack_data/attack_data.npy", images_after)  # 存储对抗样本

    # ----------------------- 对抗样本来评估模型 -----------------------
    # model_eva = keras.models.load_model("model.h5")
    # loss, acc = model_eva.evaluate(
    #     images_after.reshape(model_shape), test_labels[:count])
    # print("Restored model, accuracy: {:5.2f}%".format(100*acc))   #

    # ---------------------- 普通测试集来评估模型 --------------
    # model.summary()
    # loss, acc = model.evaluate(
    #     generate(test_images[:count], shape).reshape(model_shape), test_labels[:count])
    # print("Restored model, accuracy: {:5.2f}%".format(100*acc))   # 88.77%

    # ---------------------- 输出对普通测试集评估错误的样本 ------------------
    # pred = model.predict(test_images[:count].reshape(count, 28, 28))
    # for i in range(0, len(pred)):
    #     if(numpy.argmax(pred[i]) != test_labels[i]):
    #         print(i, " : ", numpy.max(pred), end="\t")
    #         print("pred: ", numpy.argmax(pred[i]), " actual: ", test_labels[i])

    # -------------------- 生成模型对抗样本预测的标签 ------------------------
    # model = tf.keras.experimental.load_from_saved_model("model")
    # pred = model.predict(attack_images.reshape(10000, 28, 28))
    # ls = []
    # for i in range(10000):
    #     ls.append(numpy.argmax(pred[i]))
    # print(len(ls))
    # numpy.save("attack_data/attack_pred_labels.npy", numpy.array(ls))

    # --------------- generate的核心调用测试 ----------------------
    # for i in range(0, count):
    #     img_after = attack(model, test_images[i].reshape(
    #         28*28), test_labels[i], 5, 0.3)

    #     print("\rNo.{}\tok >>>>>>>> {}{}{}\trate>>>>>>>>{:.2f}\ttime>>>>>>>>{}".format(
    #         i+1, ok, "/", count, ok/(i+1), time.time()-start_time), end='\t')
    #     print("SIM: {}".format(sim.SIM(test_images[i].reshape(
    #         28, 28), img_after.reshape(
    #         28, 28))), end="")

    # ----------------- 内存追踪 ----------------------
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # print("[ Top 20 ]")
    # for stat in top_stats[:10]:
    #     print(stat)

    # ----------------- 展示单张图片攻击前后的亚子 -----------------
    # attack_obj = 2
    # img_after = attack(model, test_images[attack_obj].reshape(
    #     28*28), test_labels[attack_obj], 5, 0.2)
    # print(sim.SIM(test_images[attack_obj].reshape(
    #     28, 28), img_after.reshape(
    #     28, 28)))
    # plt.subplot(1, 2, 1)
    # plt.imshow(test_images[attack_obj].reshape(
    #     28, 28))
    # plt.subplot(1, 2, 2)
    # plt.imshow(img_after.reshape(
    #     28, 28))
    # plt.show()
