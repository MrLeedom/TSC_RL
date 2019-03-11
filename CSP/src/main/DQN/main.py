#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Fri Mar 01 18:41:22 2019
   Description:最基本的DQN代码展示
   License: (C)Copyright 2019
'''
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

###########################   DQN    ################################
class DeepQNetwork():
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate = 0.01,
        reward_decay = 0.9,
        e_greedy = 0.9,
        replace_target_iter = 300,
        memory_size = 500,
        batch_size = 32,
        e_greedy_increment = None,
        output_graph = False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter  #隔多少步,更新一下target_net的参数
        self.memory_size = memory_size
        self.batch_size = batch_size    #神经网络在学习的时候需要用到的一个参数
        self.epsilon_increment = e_greedy_increment   #不断缩小epsilon的范围,让动作的选择更合理
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        #总的学习步数,后面注意观察其用处,epsilon根据这个来提高
        self.learn_step_counter = 0

        #初始化为空的记忆池 [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        #构造两个网络 target_net  +  eval_net
        self._build_net()
        #最初版本
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            #采用命令tensorboard --logdir=logs
            tf.summary.FileWriter('logs/', self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []   #记录误差的过程

    def _build_net(self):
        #------------------------  build evaluate_net,及时提升参数  ---------------------#
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name= 's')   #输入状态
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')   #用来计算误差loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)   #配置网络层,初始化值,collection存放存有神经网络的参数
            #调用eval_net_params这个名字就好
            #第一层, 
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            
            #第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer= b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
        with tf.variable_scope('loss'):  #求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):  #梯度下降
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)        
        
        #------------------------  build target_net,提供target Q ---------------------#
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  #下一个状态的输入
        with tf.variable_scope('target_net'):
            # t1 = tf.layers.dense(self.s_, 20, tf,nn,relu, kernel_initializer = w_initializer,
            #                     biases_initializer= b_initializer, name='t1')
            # self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
            #                     biases_initializer=b_initializer, name='t2')
            #c_names是用来存放所有变量的集合容器,是在更新target_net参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            #第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            
            #第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
    
    #记忆池的存储,从上到下,一旦存满,接着从上到下再存
    def store_transition(self, s, a, r, s_):
        #判断有没有这个属性
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        #将新的记忆代替老的记忆
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]  #observation原本一维,为了让tensorflow来处理,增加一维,变成二维

        if np.random.uniform() < self.epsilon:   #小于episilon的情况下,利用
            #提供网络输入的状态到神经网络得到每个动作的输出值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
  
    #更新网络参数
    def _replace_target_params(self):  #将e的参数更换为t
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t,e in zip(t_params, e_params)]

    def learn(self):
        #检查target网络参数是否需要被替代
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced')
        
        #从所有记忆中抽取batch部分记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)
        else:  #如果记忆库没有那么大的话,就抽取已经存下来的记忆
            sample_index = np.random.choice(self.memory_counter, size =self.batch_size)
        batch_memory = self.memory[sample_index, :]

        #接着输出两个神经网络输出的参数,获取target_net产生的Q和eval_net产生的Q
        q_next, q_eval = self.sess.run(
        [self.q_next, self.q_eval],
        feed_dict={
            self.s_ : batch_memory[:, -self.n_features:],
            self.s : batch_memory[:, :self.n_features],
        })    #传递给两个神经网络的输入参数不同
        #以下我们只是想要对应的动作的Q值,其他动作的Q值全为0,这样反向传递回去更新我们的网络参数,这部分可以关注一下
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        #训练eval_net
        _, self.cost = self.sess.run([self._train_op, self.loss],
                            feed_dict={self.s : batch_memory[:, :self.n_features],
                                    self.q_target : q_target})
        
        self.cost_his.append(self.cost)   #记录cost误差

        #逐渐增长epsilon,降低行为的随机性,这一块的系数更新放在这里可能需要修正,后期看需求!!!!!!
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
    
    #看看学习效果
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()   
        plt.savefig('./image/DQN.jpg') 
    #对于cost曲线的解释,并不是像其他神经网络那样误差一直降低,强化学习一开始是没有所有的数据,所以cost曲线也是一个比较
    #纠结的一个状态,而且收集到的数据也是参差不齐的,跟记忆池也有一定关系,在这里如果记忆池是有排序的,会不会结果更好,cost
    #曲线变化较为规律,也是一个值得研究的点.



#########################################   DQN  测试   #########################3#####################
from maze_env import Maze
def run_maze():
    step = 0
    current = 0
    for episode in range(300):
        #初始化环境观测值
        observation = env.reset()
        current += 1
        while True:
            env.render()  #刷新环境
            #强化学习智能体选择动作
            action = RL.choose_action(observation)
            #强化学习智能体根据选择的动作得到下一个状态以及对应的reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if(step > 200) and step % 5 == 0:
                RL.learn()
            
            #更换observation的值
            observation = observation_

            #何时结束一次episode
            if done:
                break
            step += 1
        print('proessing:{0}%'.format((current / 300)*100))
    print('game over')
    env.destroy()
if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(
        env.n_actions, env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000,
    )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
