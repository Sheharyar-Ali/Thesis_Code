#%%
# Theta 
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Choose file
folder = "Heli_Sim/Assets/StreamingAssets/Data/"
filename = folder + "export_leam160,0723343,7331_theta_140.csv"
file = open(filename)

# Read file
Time=[]
controlTheta = []
ffTheta =[]
controlInput = []
actualPitch = []
csvreader = csv.reader(file)
for row in csvreader:
    if(row[0] != "Time"):
        Time.append(float(row[0]))
        controlTheta.append(float(row[1]))
        ffTheta.append(float(row[2]))
        controlInput.append(float(row[3]))
        actualPitch.append(float(row[4]))
print("Done reading")
print("Time length ", len(Time))
Time = np.array(Time)
controlInput = np.array(controlInput)
controlTheta = np.array(controlTheta) 
ffTheta = np.array(ffTheta)
actualPitch = np.array(actualPitch)
error = np.zeros_like(Time) - actualPitch
thetaDf = pd.DataFrame({'Time':Time,'error':error})
thetaDf.index = Time
thetaDfTrunc = thetaDf.truncate(before=thetaDf[Time>=30].iloc[0,0])
rmse = np.sqrt(np.sum(thetaDfTrunc["error"]**2) / len(thetaDfTrunc["error"]))
print(rmse)
file.close()

# Load forcing function
filename = "Heli_Sim/Assets/Scripts/forcing_func_theta.csv"
file = open(filename)
ff=[]
t= np.arange(0,150,0.01)
csvreader = csv.reader(file)
for row in csvreader:
    if(row[1] != "forcing function"):
        ff.append(float(row[1]))
print("Done reading")
file.close()

# Plot results
plt.figure(1)
plt.title("Theta")
plt.plot(Time,controlTheta,label="CV")
plt.plot(Time,actualPitch,label="actual pitch")

plt.legend()



plt.figure(2)
plt.title("Error")
plt.plot(Time,error)
# plt.plot(t,np.abs(ffNormalised[:-1] - ffChange))

# %%
