#%%
import csv
import matplotlib.pyplot as plt
import numpy as np
filename = "Heli_Sim/Assets/Scripts/Data/export_sbinali_test_theta_140.csv"
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
rmse = np.sqrt(np.sum(error**2) / len(error))
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
plt.title("Velocities")
plt.plot(Time,controlVelocity,label="CV")
plt.plot(Time,ffVelocity,label="FF")
plt.plot(t,ff,label="actual")
# plt.plot(Time,heliVelocity,label="total")
plt.legend()

plt.figure(2)
plt.title("Error")
plt.plot(Time,error)
# %%
