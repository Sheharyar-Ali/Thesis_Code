#%%
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
folder = "Heli_Sim/Assets/StreamingAssets/Data/"
filename = folder + "export_hrte166,2115351,1547552,2052_actual_140.csv"
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
file.close()

Time = np.array(Time)
controlInput = np.array(controlInput)
controlVelocity = np.array(controlVelocity)
ffVelocity = np.array(ffVelocity)
heliVelocity = controlVelocity + ffVelocity
error = np.zeros_like(Time) - heliVelocity
uDf = pd.DataFrame({'Time':Time,'error':error})
uDf.index = Time
uDfTrunc = uDf.truncate(before=uDf[Time>=30].iloc[0,0])
rmse = np.sqrt(np.sum(uDfTrunc["error"]**2) / len(uDfTrunc["error"]))
print(rmse)




filename = "Heli_Sim/Assets/Scripts/forcing_func.csv"
file = open(filename)
ff=[]
t= np.arange(0,150,0.01)
csvreader = csv.reader(file)
for row in csvreader:
    if(row[1] != "forcing function"):
        ff.append(float(row[1]))
print("Done reading")
file.close()

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
