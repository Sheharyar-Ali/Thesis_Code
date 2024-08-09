#%%
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
filename = "Heli_Sim/Assets/Scripts/Data/export_pracc169,4549354,0737650,1796882,81981113,4561348,991578,856_actual_140.csv"
file = open(filename)
Time=[]
controlVelocity = []
ffVelocity =[]
controlInput = []
csvreader = csv.reader(file)
for row in csvreader:
    if(row[0] != "Time"):
        Time.append(float(row[0]))
        controlVelocity.append(float(row[1]))
        ffVelocity.append(float(row[2]))
        controlInput.append(float(row[3]))
print("Done reading")
print("Time length ", len(Time))
Time = np.array(Time)
controlInput = np.array(controlInput)
controlVelocity = np.array(controlVelocity)
ffVelocity = np.array(ffVelocity)
heliVelocity = controlVelocity + ffVelocity
error = np.zeros_like(Time) - heliVelocity
uDf = pd.DataFrame({'Time':Time,'error':error})
uDf.index = Time
uDf.truncate(before=uDf[Time>=30].iloc[0,0])
rmse = np.sqrt(np.sum(uDf["error"]**2) / len(uDf["error"]))
print(rmse)
file.close()
filename = "Heli_Sim/Assets/Scripts/forcing_func.csv"
file = open(filename)
ff=[]
t= np.arange(0,150,0.1)
csvreader = csv.reader(file)
for row in csvreader:
    if(row[1] != "forcing function"):
        ff.append(float(row[1]))
print("Done reading")
file.close

plt.figure(1)
plt.title("Velocities")
plt.plot(Time,controlVelocity,label="CV")
plt.plot(Time,ffVelocity,label="FF")
plt.plot(t,ff,label="actual")
# plt.plot(Time,heliVelocity,label="total")
plt.legend()

plt.figure(2)
plt.title("Error")
plt.plot(Time,error)

#%%
filename = "Heli_Sim/Assets/Scripts/Data/export_pracc188,0676_theta_140.csv"
file = open(filename)
Time=[]
controlTheta = []
ffTheta =[]
controlInput = []
csvreader = csv.reader(file)
for row in csvreader:
    if(row[0] != "Time"):
        Time.append(float(row[0]))
        controlTheta.append(float(row[1]))
        ffTheta.append(float(row[2]))
        controlInput.append(float(row[3]))
print("Done reading")
print("Time length ", len(Time))
Time = np.array(Time)
controlInput = np.array(controlInput)
controlTheta = np.array(controlTheta) 
ffTheta = np.array(ffTheta)
heliTheta = controlTheta + ffTheta
error = np.zeros_like(Time) - heliTheta
thetaDf = pd.DataFrame({'Time':Time,'error':error})
thetaDf.index = Time
print(thetaDf[Time>=30])
#thetaDf.truncate(before=thetaDf[Time>=30].iloc[0,0])
rmse = np.sqrt(np.sum(thetaDf["error"]**2) / len(thetaDf["error"]))
print(rmse)
file.close()
filename = "Heli_Sim/Assets/Scripts/forcing_func_theta.csv"
file = open(filename)
ff=[]
t= np.arange(0,150,0.1)
csvreader = csv.reader(file)
for row in csvreader:
    if(row[1] != "forcing function"):
        ff.append(float(row[1]))
print("Done reading")
file.close

plt.figure(1)
plt.title("Theta")
plt.plot(Time,controlTheta,label="CV")
plt.plot(Time,ffTheta,label="FF")
plt.plot(t,ff,label="actual")
# plt.plot(Time,heliVelocity,label="total")
plt.legend()

plt.figure(2)
plt.title("Error")
plt.plot(Time,error)
# %%
