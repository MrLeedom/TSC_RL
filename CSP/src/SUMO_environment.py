# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:19:27 2018

@author: LJH
"""

#SUMO环境代码
import os, sys
# if 'SUMO_HOME' in os.environ:
#The path of SUMO-tools to get the traci library
# # we need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     print(tools)
#     sys.path.append(tools)

# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")
sys.path.append('/usr/share/sumo/tools')
import traci
import traci.constants as tc
import numpy as np
import datetime

#Environment Model
sumoBinary = "/usr/share/sumo/sumo-0.32.0/sumo"
sumoConfig = "/work/trafficSignalControl/config/shixin_shanyin.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]  #The path to the sumo.cfg file

#reset the environment
#重置环境
def reset():
    print(sumoCmd)
    traci.start(sumoCmd)
    print('success')
    tls = traci.trafficlight.getIDList() #返回场景中所有交通灯的id列表
    return tls
reset() 
#get the starting state
def state():
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED)) #将车辆的信息订阅上，此处可以订阅一些其他的值
    p = traci.vehicle.getSubscriptionResults() #获取订阅值（即获取所有车辆的位置和速度值）
    p_state = np.zeros((60,60,2)) #这是一个60*60的状态矩阵，最后的2表示位置的bool变量以及速度两个值
    for x in p: #依次获取每一辆车的位置和速度，p是一个数组
        ps = p[x][tc.VAR_POSITION] #获取车辆i的2d坐标（x,y）,ps[0]是x坐标ps[1]是y坐标
        spd = p[x][tc.VAR_SPEED] #获取车辆i的速度
        p_state[int(ps[0]/5), int(ps[1]/5)] = [1, int(round(spd))] #把坐标等比缩小5倍作为车辆状态矩阵中的位置，1表示状态矩阵的这个位置有车辆存在，第二个值表示车辆的速度
        #v_state[int(ps[0]/5), int(ps[1]/5)] = spd
    p_state = np.reshape(p_state, [-1, 3600, 2]) #为何要重组为3600*2的数组？
    return p_state #, v_state]

#get the legal actions at the current phases of the traffic light
def getLegalAction(phases):
    legal_action = np.zeros(9)-1 #长度为9的-1数组，动作空间大小为9？
    i = 0
    for x in phases: #phases是一个长度为4的list,初始化值为[20,20,20,20]
        #if x>5: #如果这个相位时间大于5，那么对应的行为i的值为自身的序号。（因为小于5s就不能继续执行-5s的动作了。但是这里需要修改，因为要考虑最小绿灯时间。）
        if x>15:    
            legal_action[i] = i
        if x<60: #如果小于60，对应行为i+5的值也等于序号。（设60s为最大绿灯时间，大于60s就不能继续执行+5s的动作）
            legal_action[i+5] = i+5
        i +=1
    legal_action[4] = 4 #这个循环的意思是如果四个相位的时间都在5-60间，legal_action就是[0,1,2...7,8]这样一个序列，如果有一个不在区间内，那么对应的i和i+5的值就是-1(既不能执行该动作)
    return legal_action
    
#get the new phases after taking action from the current phases
def getPhaseFromAction(phases, act):
    if act<4: #如果选择的动作序号小于4，那对应的动作相位时间-5s
        phases[int(act)] -= 5
    elif act>4: #如果选择的动作序号大于4，那么对应的动作相位时间+5s
        phases[int(act)-5] += 5
    return phases

#the process of the action
#input: traffic light; new phases; waiting time in the beginning of this cycle
#output: new state; reward; End or not(Bool); new waiting time at the end of the next cycle
def action(tls, ph, wait_time):  #parameters: the phase duration in the green signals,执行动作的函数
    tls_id = tls[0] #获取当前交通灯的id，因为单交叉口只有一个所以取0
    init_p = traci.trafficlights.getPhase(tls_id) #获取交通灯的当前处于第几相位？
    prev = -1
    changed = False
    current_phases = ph #当前四个相位对应的时间
    p_state = np.zeros((60,60,2)) #设置一个60*60*2的初始零矩阵表示交通状态矩阵
    wait_time_map = {} #我增加的变量，因为91行的变量未定义

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0: #如果交通网络中的车辆数量大于0
        c_p = traci.trafficlights.getPhase(tls_id) #获取交通灯的当前相位情况
        if c_p != prev and c_p%2==0: #如果当前相位序号不是前一个相位且为序号偶数
            traci.trafficlights.setPhaseDuration(tls_id, ph[int(c_p/2)]-0.5) #设置当前相位的持续时间为相位序号int(c_p/2)对应时长再-0.5后的时间
            prev = c_p #prev表示前一个相位？
        if init_p != c_p: #如果初始相位与当前相位不同，表示相位已经改变
            changed = True
        if changed:
            if c_p == init_p: #如果当前相位等于初始相位，表示一个周期结束？
                break
        traci.simulationStep() #模拟1s
        step += 1
        if step%10==0: #每10步获取每个车辆的累计等待时间（一定时间间隔内的）
            for veh_id in traci.vehicle.getIDList():
                wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCUMULATED_WAITING_TIME))
    p = traci.vehicle.getSubscriptionResults() #获获取订阅值（即获取所有车辆的位置和速度值，以及累计等待时间）
    
    
    wait_temp = dict(wait_time_map)
    for x in p:
        ps = p[x][tc.VAR_POSITION]
        spd = p[x][tc.VAR_SPEED]
        p_state[int(ps[0]/5), int(ps[1]/5)] = [1, int(round(spd))]

    wait_t = sum(wait_temp[x] for x in wait_temp) #计算所有车辆的累计等待时间的和
    
    d = False
    if traci.simulation.getMinExpectedNumber() == 0: #判断路网中的车辆是否为空
        d = True
        
    r = wait_time-wait_t #对比前后两次的累计等待时间的变化（即奖励值）
    p_state = np.reshape(p_state, [-1, 3600, 2]) #将状态矩阵展平为一维矩阵
    return p_state,r,d,wait_t #返回状态矩阵，奖励值，是否最终状态，本次的累计等待总时间

#close the environment after every episode
def end():
    traci.close()