import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def readFileU(filename):
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
    return rmse

def readFileTheta(filename):
    file = open(filename)
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
    return rmse

def bulkRun(pNumber,fovRange,thetaRange):
    readFolder = "Heli_Sim/Assets/Scripts/Data/"+pNumber+"/"
    

    filesU = []
    filesTheta = []

    for i,angle in enumerate(fovRange):
        buffer = []
        name = "actual_" + str(angle)+".csv"
        for fileName in os.listdir(readFolder):
            if(fileName.endswith(name)):
                buffer.append(readFolder+fileName)
        filesU.append(buffer)
    for i,angle in enumerate(thetaRange):
        buffer = []
        name = "theta_" + str(angle)+".csv"
        for fileName in os.listdir(readFolder):
            if(fileName.endswith(name)):
                buffer.append(readFolder+fileName)
        filesTheta.append(buffer)

    # Read u files
    rmseListFull =[]
    rmseListAverage=[]
    for i,files in enumerate(filesU):
        buffer=[]
        for file in files:
            error = readFileU(file)
            buffer.append(error)
        rmseListFull.append(buffer)
        rmseListAverage.append(np.average(np.array(buffer)))
    
    # Read theta files
    rmseListFullTheta =[]
    rmseListAverageTheta=[]
    for i,files in enumerate(filesTheta):
        buffer=[]
        for file in files:
            error = readFileTheta(file)
            buffer.append(error)
        rmseListFullTheta.append(buffer)
        rmseListAverageTheta.append(np.average(np.array(buffer)))
    print(rmseListFullTheta)
    print(rmseListAverageTheta)
    return rmseListAverage,rmseListFull,rmseListAverageTheta,rmseListFullTheta

fovRange = [20,30,60,90,120,140]
thetaRange = [20,140]
writeFolder = "Visuals/Results/"
pNumbers = ["P1"]
figure = 1
for pNumber in pNumbers:
    AverageU,_,AverageTheta,_ = bulkRun(pNumber,fovRange,thetaRange)
    saveNameU = writeFolder + "RMSE_U_" + pNumber
    saveNameTheta = writeFolder + "RMSE_Theta_" + pNumber
    
    uFig = plt.figure(figure)
    figure+=1
    plt.title("Change in RMSE for velocity task for "+pNumber)
    plt.ylabel("RMSE [(m/s) ^ 2]")
    plt.xlabel("Field of View")
    plt.plot(fovRange,AverageU,marker="x")
    plt.savefig(saveNameU)
    
    ThetaFig = plt.figure(figure)
    figure+=1
    plt.title("Change in RMSE for Theta task for "+pNumber)
    plt.ylabel("RMSE [(deg) ^ 2]")
    plt.xlabel("Field of View")
    plt.plot(thetaRange,AverageTheta,marker="x")
    plt.savefig(saveNameTheta)
plt.show()