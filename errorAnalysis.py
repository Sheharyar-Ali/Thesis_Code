#%%
import csv
import matplotlib.pyplot as plt
import numpy as np
filename = "Heli_Sim/Assets/Scripts/export_sbinali_test_actual_20.csv"
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
Time = np.array(Time[300:])
controlInput = np.array(controlInput[300:])
controlVelocity = np.array(controlVelocity[300:])
ffVelocity = np.array(ffVelocity[300:])
heliVelocity = controlVelocity + ffVelocity
error = np.zeros_like(Time) - heliVelocity
rmse = np.sqrt(np.sum(error**2) / len(error))
print(rmse)

file.close()
plt.figure(1)
plt.title("Velocities")
plt.plot(Time,controlVelocity,label="CV")
plt.plot(Time,ffVelocity,label="FF")
# plt.plot(Time,heliVelocity,label="total")
plt.legend()

plt.figure(2)
plt.title("Error")
plt.plot(Time,error)
# %%
