import os, sys
sys.path.append('D:/SUMO/tools')
import traci
import traci.constants as tc
import numpy as np
import datetime
import matplotlib.pyplot as plt

#Environment Model
sumoBinary = "D:/SUMO/bin/sumo"
sumoConfig = "C:/Users/Administrator/Desktop/code/shixin_shanyin_7/shixin_shanyin.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]  #The path to the sumo.cfg file
traci.start(sumoCmd)
tls = traci.trafficlight.getIDList()
dets=traci.lanearea.getIDList()
tls_id = tls[0]
init_p = traci.trafficlight.getPhase(tls_id)
prev = -1
changed = False
ph = [60,60,60,60]
step = 0
zq_step=0
queue_max = []
queue_sum = []
q_plot_i = []
r_plot_i = []
for _ in range(371):
    traci.simulationStep()
while zq_step < 51: #如果交通网络中的车辆数量大于0
    c_p = traci.trafficlights.getPhase(tls_id) #获取交通灯的当前相位情况
    if c_p != prev and c_p%2==0: #如果当前相位序号不是前一个相位且为序号偶数
        if changed==True:
            queue_length=[]
            for det in dets:
                queue=traci.lanearea.getJamLengthVehicle(det)
                queue_length.append(queue)
            queue_max.append(max(queue_length))
            queue_sum.append(sum(queue_length))#/len(queue_length)
            print(queue_length)
        traci.trafficlight.setPhaseDuration(tls_id, ph[int(c_p/2)]-0.5) #设置当前相位的持续时间为相位序号int(c_p/2)对应时长再-0.5后的时间
        prev = c_p #prev表示前一个相位？
        sec=0
    if init_p != c_p: #如果初始相位与当前相位不同，表示相位已经改变
        changed = True
    if sec>25 and c_p==0:
        flag = 0
        for det in ['gneE0_1', 'gneE0_2', 'gneE0_3', 'gneE0_4','gneE2_1', 'gneE2_2', 'gneE2_3', 'gneE2_4']:
            if traci.lanearea.getJamLengthVehicle(det)>0:
                flag = 1
        if flag == 0:
            traci.trafficlight.setPhase(tls_id,1)
    if sec>25 and c_p==2:
        flag = 0
        for det in ['gneE2_5', 'gneE0_5']:
            if traci.lanearea.getJamLengthVehicle(det)>0:
                flag = 1
        if flag == 0:
            traci.trafficlight.setPhase(tls_id,3)
    if sec>25 and c_p==4:
        flag = 0
        for det in ['gneE1_1', 'gneE1_2','gneE3_1', 'gneE3_2']:
            if traci.lanearea.getJamLengthVehicle(det)>0:
                flag = 1
        if flag == 0:
            traci.trafficlight.setPhase(tls_id,5)
    if sec>25 and c_p==6:
        flag = 0
        for det in ['gneE1_3', 'gneE3_3']:
            if traci.lanearea.getJamLengthVehicle(det)>0:
                flag = 1
        if flag == 0:
            traci.trafficlight.setPhase(tls_id,7)
    if changed:
        if c_p == init_p: #如果当前相位等于初始相位，表示一个周期结束？
            # print(queue_max)
            # print(queue_sum)
            r = min(queue_max) - max(queue_max)
            q_m = sum(queue_sum) / len(queue_sum)
            r_plot_i.append(r)
            q_plot_i.append(q_m)
            queue_max=[]
            queue_sum = []
            changed = False
            zq_step += 1
            step=0
    traci.simulationStep() #模拟1s
    sec += 1
    step += 1
traci.close()
print('50个周期内平均排队长度为',sum(q_plot_i[1:])/len(q_plot_i[1:]))
print('50个周期内平均奖励值为',sum(r_plot_i)/len(r_plot_i))
fig=plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.ylabel('平均排队长度（辆）',{'size':15})
plt.xlabel('周期数',{'size':15})
plt.tick_params(labelsize=15)
plt.plot(np.arange(len(q_plot_i[1:])), q_plot_i[1:],color='black')
plt.show()
fig=plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.ylabel('奖励值',{'size':15})
plt.xlabel('周期',{'size':15})
plt.tick_params(labelsize=15)
plt.plot(np.arange(len(r_plot_i)), r_plot_i,color='black')
plt.show()