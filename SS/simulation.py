# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 18:34:15 2018

@author: LJH
"""
from RLbrain import Qnetwork
from P_experience_replay import priorized_experience_buffer
import SUMO_environment as sumo
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

#参数
#batch_size = 128 #每个训练步需要的经验数量.
update_freq = 1 #How often to perform a training step.
y = .99 #折扣系数
startE = 1 #初始的epsilon
endE = 0.01 #最终的epsilon
#anneling_steps = 10000. #epsilon衰减需要的步数.
#num_episodes = 1500#000 #整个环境的迭代步数.
#pre_train_steps = 2000#0000 #开始训练之前随机选择动作状态的总步数
#max_epLength = 500 #episode的最大长度
batch_size = 64
anneling_steps = 2000. #新设定
num_episodes = 201#新的设定（上面是原设定）
pre_train_steps = 500 #新的设定（上面是原设定）
max_epLength = 50 #新的设定（上面是原设定）
load_model = False #是否要载入一个已有的模型.
action_num =  9 #动作空间大小
path = "C:/Users/Administrator/Desktop/code/models" #储存模型的路径.
h_size = 64 #最后一层卷积层的大小.
tau = 0.001 #更新目标网络的速率

#用于绘图的值
loss_plot=[]
r_plot=[]
q_plot=[]

tf.reset_default_graph()
#define the main QN and target QN
mainQN = Qnetwork(h_size,np.int32(action_num))
targetQN = Qnetwork(h_size,np.int32(action_num))

init = tf.global_variables_initializer()
saver = tf.train.Saver()  
#经常需要在训练完一个模型之后保存训练结果,这些结果是指模型的参数,以便下次迭代的训练或者用作测试
#Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

#define the memory
myBuffer0 = priorized_experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

#create lists to contain total rewards and steps per episode
jList = []  #number of steps in one episode
rList = []  #reward in one episode
wList = []  #the total waiting time in one episode
awList = []  #the average waiting time in one episode
tList = []   #thoughput in one episode (number of generated vehicles)
nList = []   #stops' percentage (number of stopped vehicles divided by the total generated vehicles)
total_steps = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

init_phases = [45,25,25,25]#初始相位时间
#它能让你在运行图的时候,插入一些计算图,这些图是由某些操作构成的,这对于工作在交互环境中的人们来说非常便利
sess = tf.InteractiveSession()

#record the loss,用来显示标量信息,一般在画loss和accuracy时会用到这个函数
tf.summary.scalar('Loss', mainQN.loss)

rfile = open(path+'/reward-rl.txt', 'w')
wfile = open(path+'/wait-rl.txt', 'w')
awfile = open(path+'/acc-wait-rl.txt', 'w')
tfile = open(path+'/throput-rl.txt', 'w')
r_output = open('C:/Users/Administrator/Desktop/code/rewardoutput7.csv','a+')
q_output = open('C:/Users/Administrator/Desktop/code/queueoutput7.csv','a+')


merged = tf.summary.merge_all()
s_writer = tf.summary.FileWriter(path+'/train', sess.graph)
s_writer.add_graph(sess.graph)
    
sess.run(init)
tf.global_variables_initializer().run()
if load_model == True:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess,ckpt.model_checkpoint_path)
updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.


#所有episode的运行过程
for i in range(1,num_episodes):
    episodeBuffer0 = priorized_experience_buffer()
    #Reset environment and get first new observation
    tls = sumo.reset()
    s = sumo.state()#np.random.rand(1,10000)
    current_phases = list(init_phases)
    wait_time_map = {}
    wait_time = 0 #我添加的，因为wait_time没有定义
    d = False
    rAll = 0
    j = 0
    r_plot_i=[]
    q_plot_i=[]
    
    print("III:", i, e)  #衰减率
    while j < max_epLength:
        j+=1
        
        #get the legal actions at the current state
        legal_action = sumo.getLegalAction(current_phases) #np.random.randint(1,action_num,size=action_num) #[1,2,-1,4,5]
        print(current_phases)
        
        #Choose an action (0-8) by greedily (with e chance of random action) from the Q-network
        #用epsilon选择动作的策略
        if np.random.rand(1) < e or total_steps < pre_train_steps:#如果随机值小于e或者总步数小于预训练的步数，就随机选择动作
            a_cnd = [x for x in legal_action if x!=-1] #选择不是-1的所有动作
            a_num = len(a_cnd) #剩余动作的长度
            a = np.random.randint(0, a_num) #在0-动作数量之间随机选择一个数字
            a = a_cnd[a] #选择对应的动作
        else:
            #np.reshape(s, [-1,3600,2])
            np.reshape(s, [-1,600,2])
            #加大这个不合理程度
            legal_a_one = [0 if x!=-1 else -99999 for x in legal_action] #非-1的动作对应的值为0，-1的动作对应的值为一个很大的负数
            a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:s,mainQN.legal_actions:[legal_a_one]})[0] #返回预测动作的值
        
        ph = sumo.getPhaseFromAction(current_phases,a) #这一步应该是根据动作来修改当前相位时间并返回修改后的相位        
        s1, r, d, wait_time,q_m = sumo.getNextStep(tls, ph, wait_time) #获得下一个状态、奖励、是否结束、以及等待时间
        current_phases = ph #更新当前的相位时间配置

        total_steps += 1
        legal_a_one = [0 if x!=-1 else -99999 for x in legal_action] #the penalized Q value for illegal actions
        legal_act_s1 = sumo.getLegalAction(ph)#下一个交通相位中的可选行为
        legal_a_one_s1 = [0 if x!=-1 else -99999 for x in legal_act_s1] #legal_a_one和legal_a_one_s1分别是前一轮相位配置和当前相位配置对应动作空间的值
        episodeBuffer0.add(np.reshape(np.array([s,a,r,s1,d,legal_a_one, legal_a_one_s1]),[1,7])) #Save the experience to our episode buffer.

        if total_steps > pre_train_steps:
            if e > endE:
                e -= stepDrop
            if total_steps % (update_freq) == 0:
                trainBatch = myBuffer0.priorized_sample(batch_size) #Get a random batch of experiences.
                #trainBatch = episodeBuffer0.priorized_sample(batch_size)
                indx = np.reshape(np.vstack(trainBatch[:,1]), [batch_size])
                indx = indx.astype(int)
                trainBatch = np.vstack(trainBatch[:,0]) #将经验矩阵转置，变成batch_size行，一列的矩阵

                #Below we perform the Double-DQN update to the target Q-values 
                #action from the main QN
                Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3]),mainQN.legal_actions:np.vstack(trainBatch[:,6])}) #原来的legal_actions是第5列（感觉有问题），我现在改为第6列
                #Q value from the target QN
                Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3]), targetQN.legal_actions:np.vstack(trainBatch[:,6])})
                # get targetQ at s'
                end_multiplier = -(trainBatch[:,4] - 1)  #if end, 0; otherwise 1
                doubleQ = Q2[range(batch_size),Q1]
                targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)#y是折扣系数，这个值就是TD目标值。

                #Update the network with our target values.
                summry, err, ls, md = sess.run([merged, mainQN.td_error, mainQN.loss, mainQN.updateModel],  \
                    feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1],mainQN.legal_actions:np.vstack(trainBatch[:,5])})

                s_writer.add_summary(summry, total_steps)
                #update the target QN and the memory's prioritization
                updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
                myBuffer0.updateErr(indx, err)
                #episodeBuffer0.updateErr(indx, err)
                r_plot_i.append(r)
                loss_plot.append(ls)
                q_plot_i.append(q_m)
                print('神经网络第',total_steps-pre_train_steps,'次训练，损失值为',ls)
        rAll += r
        s = s1
        print('episode',i,"进度已完成：",int(100*j/max_epLength),'%')

        if d == True:
            break
    sumo.end()
    if len(r_plot_i)>0:
        r_plot.append(sum(r_plot_i)/len(r_plot_i))
        q_plot.append(sum(q_plot_i)/len(q_plot_i))

    #save the data into the myBuffer
    myBuffer0.add(episodeBuffer0.buffer)
    
#     jList.append(j)
#     rList.append(rAll)
#     rfile.write(str(rAll)+'\n')
#     wt = sum(wait_time_map[x] for x in wait_time_map)
#     wtAve = wt/len(wait_time_map)
#     wList.append(wtAve)
#     wfile.write(str(wtAve)+'\n')
#     awList.append(wt)
#     awfile.write(str(wt)+'\n')
#     tList.append(len(wait_time_map))
#     tfile.write(str(len(wait_time_map))+'\n')
#     tmp = [x for x in wait_time_map if wait_time_map[x] > 1]
#     nList.append(len(tmp)/len(wait_time_map))
#     print("Total Reward---------------",  rAll)
#     #Periodically save the model. 
#     if i % 100 == 0:
#         saver.save(sess,path+'/model-'+str(i)+'.cptk')
#         print("Saved Model")
# #         if len(rList) % 10 == 0:
# #             print(total_steps,np.mean(rList[-10:]), e)
saver.save(sess,path+'/model-'+str(i)+'.cptk')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
print(str(r_plot).strip('[').strip(']'),file=r_output)
print(str(q_plot).strip('[').strip(']'),file=q_output)
plt.plot(np.arange(len(r_plot)), r_plot)
plt.ylabel('平均奖励值')
plt.xlabel('训练回合数')
plt.show()
# print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")