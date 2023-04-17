import sys
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import utils
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ScalingAttack(object):
    def __init__(self, sourceImg=None, targetImg=None, **kwargs):
        _, __, *channel = sourceImg.shape
        if not channel:
            self.sourceImg = sourceImg[:, :, np.newaxis]
        else:
            self.sourceImg = sourceImg
        _, __, *channel = targetImg.shape
        if not channel:
            self.targetImg = targetImg[:, :, np.newaxis]
        else:
            self.targetImg = targetImg

        # 初始化参数
        self.params = {'func': cv2.resize,
                       'interpolation': cv2.INTER_LINEAR,
                       'L_dist': 'L2',
                       'penalty': 1.,
                       'img_factor': 255.}
        keys = self.params.keys()
        for key, value in kwargs.items():
            assert key in keys, ('Improper parameter %s, '
                                 'The parameter should in: '
                                 '%s' %(key, keys))
            self.params[key] = value

    def setResizeMethod(self, func=cv2.resize,
                        interpolation=cv2.INTER_NEAREST):
        
        #设置缩放的方法,func是缩放函数，比如cv2.resize,interpolation是缩放选项，比如cv2.INTER_NEAREST.
        self.params['func'] = func
        self.params['interpolation'] = interpolation

    def setSourceImg(self, sourceImg):
        _, __, *channel = sourceImg.shape
        if not channel:
            self.sourceImg = sourceImg[:, :, np.newaxis]
        else:
            self.sourceImg = sourceImg

    def setTargetImg(self, targetImg):
        _, __, *channel = targetImg.shape
        if not channel:
            self.targetImg = targetImg[:, :, np.newaxis]
        else:
            self.targetImg = targetImg

    def estimateConvertMatrix(self, inSize, outSize):

        #估计转换矩阵，输入参数的inSize是缩放前原输入的大小，outSiz是缩放后的大小，输出是转换矩阵

        #输入一个虚拟的测试图像(单位矩阵* 255)
        inputDummyImg = (self.params['img_factor'] *
                         np.eye(inSize)).astype('uint8')
        outputDummyImg = self._resize(inputDummyImg,
                                      outShape=(inSize, outSize))
        #将转换矩阵的元素在缩放到[0,1]
        convertMatrix = (outputDummyImg[:,:,0] /
                (np.sum(outputDummyImg[:,:,0], axis=1)).reshape(outSize, 1))

        return convertMatrix

    def _resize(self, inputImg, outShape=(0,0)):
        func = self.params['func']
        interpolation = self.params['interpolation']
        # PIL的resize只能应用于PIL.Image对象
        if func is Image.Image.resize:
            inputImg = Image.fromarray(inputImg)
        if func is cv2.resize:
            outputImg = func(inputImg, outShape, interpolation=interpolation)
        else:
            outputImg = func(inputImg, outShape, interpolation)
            outputImg = np.array(outputImg)
        if len(outputImg.shape) == 2:
            outputImg = outputImg[:,:,np.newaxis]
        return np.array(outputImg)

    def _getPerturbationGPU(self, convertMatrixL, convertMatrixR, source, target):
        #为从source image到target image的缩放攻击生成扰动，输入分别为原向量，shape为（n,1）和目标向量,shape为（m,1）
        #返回的是扰动向量，shape为(n,1)
        penalty_factor = self.params['penalty']
        p, q, c = source.shape
        a, b, c = target.shape
        convertMatrixL = tf.constant(convertMatrixL, dtype=tf.float32)
        convertMatrixR = tf.constant(convertMatrixR, dtype=tf.float32)
        modifier_init = np.zeros(source.shape)
        source = tf.constant(source, dtype=tf.float32)
        target = tf.constant(target, dtype=tf.float32)
        modifier = tf.Variable(modifier_init, dtype=tf.float32)
        modifier_init = None
        attack = (tf.tanh(modifier) + 1) * 0.5
        x = tf.reshape(attack, [p, -1])
        x = tf.matmul(convertMatrixL, x)
        x = tf.reshape(x, [-1, q, c])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [q, -1])
        x = tf.matmul(convertMatrixR, x)
        x = tf.reshape(x, [-1, a, c])
        output = tf.transpose(x, [1, 0, 2])
        delta_1 = attack - source
        delta_2 = output - target
        # OpenCV - GRB
        obj1 = tf.reduce_sum(tf.square(delta_1)) / (p*q)
        obj2 = penalty_factor * tf.reduce_sum(tf.square(delta_2)) / (a*b)
        obj = obj1 + obj2
        max_iteration = 3000
        with tf.compat.v1.Session() as sess:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
            op = optimizer.minimize(obj, var_list=[modifier])
            sess.run(tf.compat.v1.global_variables_initializer())
            prev = np.inf
            for i in range(max_iteration):
                _, obj_value = sess.run([op, obj])
                if i % 1000 == 0:
                    print(obj_value)
                    if obj_value > prev*0.999:
                        break
                    prev = obj_value
            attack_opt = attack.eval()
            print("Obj1:", obj1.eval(), ", Obj2:", obj2.eval())
        return attack_opt

    def attack(self):     
       #发动攻击，返回attack image
        sourceImg = self.sourceImg
        targetImg = self.targetImg
        sourceHeight, sourceWidth, sourceChannel = sourceImg.shape
        targetHeight, targetWidth, targetChannel = targetImg.shape
        convertMatrixL = self.estimateConvertMatrix(sourceHeight, targetHeight)
        convertMatrixR = self.estimateConvertMatrix(sourceWidth, targetWidth)
        img_factor = self.params['img_factor']
        sourceImg = sourceImg / img_factor
        targetImg = targetImg / img_factor
        source = sourceImg
        target = targetImg
        self.info()
        attackImg = self._getPerturbationGPU(convertMatrixL,
                                             convertMatrixR,
                                             source, target)

        print(np.max(attackImg))
        print(np.min(attackImg))
        print('Done! :)')
        return np.uint8(attackImg * img_factor)

    def info(self):

        if self.params['func'] is cv2.resize:
            func_name = 'cv2.resize'
            inter_dict = ['cv2.INTER_NEAREST',
                          'cv2.INTER_LINEAR',
                          'cv2.INTER_CUBIC',
                          'cv2.INTER_AREA',
                          'cv2.INTER_LANCZOS4']
            inter_name = inter_dict[self.params['interpolation']]
        elif self.params['func'] is Image.Image.resize:
            func_name = 'PIL.Image.resize'
            inter_dict= ['PIL.Image.NEAREST',
                         'PIL.Image.LANCZOS',
                         'PIL.Image.BILINEAR',
                         'PIL.Image.BICUBIC']
            inter_name = inter_dict[self.params['interpolation']]

        sourceShape = (self.sourceImg.shape[1],
                       self.sourceImg.shape[0],
                       self.sourceImg.shape[2])

        targetShape = (self.targetImg.shape[1],
                       self.targetImg.shape[0],
                       self.targetImg.shape[2])

        print('------------------------------------')
        print('**********|Scaling Attack|**********')
        print('Source image size: %s' %str(sourceShape))
        print('Target image size: %s' %str(targetShape))
        print()
        print('Resize method: %s' %func_name)
        print('interpolation: %s' %inter_name)
        print('------------------------------------')


def test():
    sourceImgPath = sys.argv[1]
    targetImgPath = sys.argv[2]
    attackImgPath = sys.argv[3]

    sourceImg = utils.imgLoader(sourceImgPath)
    targetImg = utils.imgLoader(targetImgPath)

    print("Source image: %s" %sourceImgPath)
    print("Target image: %s" %targetImgPath)

    sc_gpu = ScalingAttack(sourceImg,
                  targetImg,
                  func=cv2.resize,
                  interpolation=cv2.INTER_LINEAR,
                  penalty=1,
                  img_factor=255.)

    attackImg = sc_gpu.attack()
    utils.imgSaver(attackImgPath, attackImg)
    print("The attack image is saved as %s" %attackImgPath)

if __name__ == '__main__':
    test()
