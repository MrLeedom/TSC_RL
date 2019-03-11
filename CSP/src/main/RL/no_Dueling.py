#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Thu Mar 07 09:20:52 2019
   Description:3DQN+CNN结构
   License: (C)Copyright 2019
'''
#The class of dueling DQN with three convolutional layers
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plt
import scipy.misc
import os
import scipy.stats as ss
import math
import collections

np.random.seed(1)
tf.set_random_seed(1)
#神经网络
class Qnetwork():
    def __init__(self,h_size,action_num):
        #The network recieves a state from the sumo, flattened into an array.
        #从sumo获取一个状态，展平成一个数组,60行 * 60列
        #It then resizes it and processes it through three convolutional layers.
        #修改输入的结构后，通过三个卷积层进行处理
        self.scalarInput =  tf.placeholder(shape=[None,600,2],dtype=tf.float32) #输入
        self.legal_actions =  tf.placeholder(shape=[None,action_num],dtype=tf.float32) #行为空间
        
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,20,30,2])  #把输入的结构修改
        #slim.conv2d自带卷积功能+激活函数,其中slim可以使建立\训练和评估神经网络更加简单
        #第一层卷积层c，卷积核个数（即输出）32，大小为8*8，步长为4
        with tf.variable_scope('cov_1'):
            self.conv1 = slim.conv2d( \
                inputs=self.imageIn,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID', activation_fn=tf.nn.relu, biases_initializer=None)
        with tf.variable_scope('cov_2'):
            self.conv2 = slim.conv2d( \
                inputs=self.conv1,num_outputs=64,kernel_size=[2,2],stride=[1,1],padding='VALID', activation_fn=tf.nn.relu, biases_initializer=None)
        with tf.variable_scope('cov3'):
            self.conv3 = slim.conv2d( \
                inputs=self.conv2,num_outputs=128,kernel_size=[2,2],stride=[1,1],padding='VALID', activation_fn=tf.nn.relu, biases_initializer=None)
        '''
        #参数说明:
            inputs 需要做卷积的图像
            num_outputs 指定卷积核的个数(就是filter的个数)
            kernel_size 用于指定卷积核的维度(卷积核的宽度,卷积核的高度)
            stride 为卷积时在图像每一维的步长
            padding 为padding方式选择,分成Valid和same两种
            activation_fn 用于激活函数的指定,默认为relu函数
            biases_initializer 用于指定biases的初始化程序
        '''   
        #It is split into Value and Advantage
        #将输出扁平化但保留batch_size,假设第一维是batch
        self.stream = slim.flatten(self.conv3) #将conv3的输出值从三维变为1维的数组，保留batch_size，假设第一维是batch
        #前两个参数分别为网络输入\输出的神经元数量
        with tf.variable_scope('fully_connected'):
            self.stream0 = self.stream0 = slim.fully_connected(self.stream, action_num, activation_fn=tf.nn.relu)
        #The final Q value is the addition of the Q value and penelized value for illegal actions
        self.Qout = tf.add(self.stream0, self.legal_actions)  #这一块也是创新点设计,加大所谓的惩罚力度
        #The predicted action
        #预测的动作,argmax中的0:按列计算  1:按行计算
        self.predict = tf.argmax(self.Qout,1) #tf.argmax(input, axis=None, name=None, dimension=None)，axis为1则为按行取最大的那个数的序号
        
        #Below we obtain the loss by taking the mean square error between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        #将一个值化为一个概率分布的向量
        self.actions_onehot = tf.one_hot(self.actions,np.int32(action_num),dtype=tf.float32)
        #压缩求和,用于降维,axis=1表示行
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        with tf.variable_scope('loss'):
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
        with tf.variable_scope('train'):
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)
        
    # def relu(self, x, alpha=0.01, max_value=None):
    #     '''ReLU.这个函数在此处到底有没有用到,待研究
    #
    #     alpha: slope of negative section.
    #     '''
    #     negative_part = tf.nn.relu(-x)
    #     x = tf.nn.relu(x)
    #     if max_value is not None:
    #         #tf.clip_by_value(A, min, max)输出一个张量,把A中的每一个元素的值都压缩到min和max之间
    #         x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
    #                              tf.cast(max_value, dtype=tf.float32))
    #     x -= tf.constant(alpha, dtype=tf.float32) * negative_part#若为正数,则输出该数,若为负数,输出一个很小的正数,缩小一点
    #     return x
    
    def choose_action(self, epsilon, observation, legal_action, sess, flag):
        #Choose an action (0-8) by greedily (with e chance of random action) from the Q-network
        #用epsilon选择动作的策略
        if np.random.uniform() < epsilon or flag == False:                  #如果随机值小于e，就随机选择动作
            a_selected = [x for x in legal_action if x!=-1]     #选择不是-1的所有动作
            a_num = len(a_selected)                             #剩余动作的长度
            a = np.random.randint(0, a_num)                     #在0-动作数量之间随机选择一个数字
            action = a_selected[a]                              #选择对应的动作
        else:
            np.reshape(observation, [-1,600,2])
            legal_a_one = [0 if x!=-1 else -99999 for x in legal_action] #非-1的动作对应的值为0，-1的动作对应的值为一个很大的负数
            action = sess.run(self.predict,feed_dict={self.scalarInput:observation, self.legal_actions:[legal_a_one]})[0] #返回预测动作的值
        return action

