import os, sys
sys.path.append('D:/SUMO/tools')
import traci
import traci.constants as tc
import numpy as np
import datetime

#Environment Model
sumoBinary = "D:/SUMO/bin/sumo"
sumoConfig = "C:/Users/Administrator/Desktop/code/shixin_shanyin_7/shixin_shanyin.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]  #The path to the sumo.cfg file

# outputfile = open('C:/Users/Administrator/Desktop/code/output.csv','a+')

# print("phase_1,max_1,mean_1,phase_2,max_2,mean_2,phase_3,max_3,mean_3,phase_4,max_4,mean_4",file=outputfile)
# outputfile = open('C:/Users/Administrator/Desktop/code/output.csv','a+')
result = open('C:/Users/Administrator/Desktop/code/3434.csv','a+')
# print("phase_1,max_1,mean_1,phase_2,max_2,mean_2,phase_3,max_3,mean_3,phase_4,max_4,mean_4",file=outputfile)
traci.start(sumoCmd)
for _ in range(360):
    tls = traci.simulationStep() #返回场景中所有交通灯的id列表
dets = traci.lanearea.getIDList()
queue_length=[]
queue_length1=[]
for det in dets:
    queue = traci.lanearea.getJamLengthVehicle(det)
    queue1 = traci.lanearea.getLastStepHaltingNumber(det)
    queue_length.append(queue)
    queue_length1.append(queue1)
print(dets,queue_length,queue_length1,np.argmax(queue_length))
for veh_id in traci.vehicle.getIDList():
    traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))
for veh_id in traci.vehicle.getIDList():
    p = traci.vehicle.getSubscriptionResults(veh_id) #获取订阅值（即获取所有车辆的位置和速度值）
    px = p[tc.VAR_POSITION]
    sp = p[tc.VAR_SPEED]
a = [1,2,3]
print(str(a).strip('[').strip(']'),file=result)
traci.close()