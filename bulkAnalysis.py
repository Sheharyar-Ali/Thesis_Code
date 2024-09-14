import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import f_oneway

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
    readFolder = "ExpData/"+pNumber+"/"
    

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
    return rmseListAverage,rmseListFull,rmseListAverageTheta,rmseListFullTheta


def bulkAnalysis(meanErrors,fovRange):
    fovConst = sm.add_constant(fovRange)
    model = sm.OLS(meanErrors,fovConst)
    results=model.fit()
    return results.summary()

def sortOrder(sequence,id):
    if(id < len(sequence)):
        first = id -1
    else:
        first = id % len(sequence) - 1
    if(first!=0):
        slice1 = sequence[first:]
        slice2 = sequence[0:first]
        return slice1 + slice2
    else: 
        return  sequence
def sortOrderDebug(sequence,id):
    if(id <= len(sequence)):
        first = id -1
    else:
        first = id % len(sequence) - 1
    print(id,first,sequence)
    if(first!=0):
        slice1 = sequence[first:]
        slice2 = sequence[0:first]
        print(slice1,slice2)
        return slice1 + slice2
    else: 
        return  sequence

def sortRunOrder(expMatrixU,expMatrixTheta,expMatrixFull,participants):
    fullU =[]
    fullTheta=[]
    fullAll=[]
    fileListAll = []
    for i,participant in enumerate(participants):
        listU=[]
        listTheta=[]
        listFull=[]
        fileListU=[]
        fileListTheta=[]
        fileListFull=[]
        readFolder = "ExpData/"+participant+"/"
        runOrderU = sortOrder(expMatrixU,i+1)
        runOrderTheta = sortOrder(expMatrixTheta,i+1)
        runOrderFull = sortOrder(expMatrixFull,i+1)
        for cond in runOrderFull:
            fovChosen = cond[1:]
            if(cond[0] == "U"):
                name = "actual_" + str(fovChosen)+".csv"
            else:
                name = "theta_" + str(fovChosen)+".csv"
            buffer = []
            # name = "actual_" + str(cond)+".csv"
            for fileName in os.listdir(readFolder):
                if(fileName.endswith(name)):
                    buffer.append(fileName)
            if(len(buffer) > 2):
                buffer = buffer[1:]
            
            if(len(buffer[0]) > len(buffer[1])):
                order = [buffer[1],buffer[0]]   
            else:
                order = [buffer[0],buffer[1]]
            if(cond[0] == "U"):
                listU.append(readFileU(readFolder+order[0]))
                listU.append(readFileU(readFolder+order[1]))      
                fileListU.append(order[0])
                fileListU.append(order[1])
                listFull.append(readFileU(readFolder+order[0]))
                listFull.append(readFileU(readFolder+order[1]))      
                
            else:
                listTheta.append(readFileTheta(readFolder+order[0]))
                listTheta.append(readFileTheta(readFolder+order[1]))      
                fileListTheta.append(order[0])
                fileListTheta.append(order[1])
                listFull.append(readFileTheta(readFolder+order[0]))
                listFull.append(readFileTheta(readFolder+order[1]))                  
            fileListFull.append(order[0])
            fileListFull.append(order[1])
        fullU.append(listU)
        fullTheta.append(listTheta)
        fullAll.append(listFull)
        fileListAll.append(fileListFull)
    return fullU,runOrderU,fullTheta,runOrderTheta,fullAll,fileListAll
    
fovRange = [20,30,60,90,120,140]
thetaRange = np.array([20,140])
fullList = ["U20","U30","U60","U90","U120","U140","T20","T140"]
lowFOV = np.array([20,30,60,90])
highFOV = np.array([90,120,140])
writeFolder = "Visuals/Results/"
pNumbers = ["VEOR","AGES","ILAH","AVUN","AKLA","ENIE","AAOO","HRTE","ARAM","UGIN","LADV","LEAM"]
numberParticipants = 14
runOrder = np.arange(1,17)
partList = np.arange(1,numberParticipants+1)
fullAverageU=[]
fullAverageTheta=[]
lowFOVAverageU=[]
highFOVAverageU=[]
figure = 1


_,_,_,_,fullAll,runOrderFull= sortRunOrder(fovRange,thetaRange,fullList,pNumbers)
print(np.array(runOrderFull))
print(np.array(fullAll))

runOrderFig = plt.figure(figure)
figure+=1
axRunOrder = runOrderFig.add_subplot(111)
plt.title("RMSE perfromance vs run order")
plt.ylabel("RMSE")
plt.xlabel("Run number")
for i,participant in enumerate(fullAll):
    axRunOrder.plot(runOrder,np.array(participant),label=i+1,marker="x")
plt.legend()
plt.show()

exit()


bulkFigU = plt.figure(figure)
figure+=1
ax = bulkFigU.add_subplot(111)
plt.title("Change in RMSE for velocity task for all participants")
plt.ylabel("RMSE [(m/s) ^ 2]")
plt.xlabel("Field of View")

bulkFigTheta = plt.figure(figure)
figure+=1
axTheta = bulkFigTheta.add_subplot(111)
plt.title("Change in RMSE for Theta task for all participants")
plt.ylabel("RMSE [(deg) ^ 2]")
plt.xlabel("Field of View")

for pNumber in pNumbers:
    AverageU,_,AverageTheta,_ = bulkRun(pNumber,fovRange,thetaRange)
    fullAverageU.append(AverageU)
    fullAverageTheta.append(AverageTheta)
    lowFOVAverageU.append(AverageU[0:4])
    highFOVAverageU.append(AverageU[3:])
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
    
    ax.plot(fovRange,AverageU,marker="x",label= pNumber)
    axTheta.plot(thetaRange,AverageTheta,marker="x",label=pNumber)
    plt.close(uFig)
    plt.close(ThetaFig)
ax.legend()
axTheta.legend()
bulkFigU.savefig(writeFolder+"RMSE_All_U")
bulkFigTheta.savefig(writeFolder+"RMSE_All_Theta")
plt.close('all')

byFOVU = [[],[],[],[],[],[]]
ULabels = ["FoV 20", "FoV 30", "FoV 60", "FoV 90", "FoV 120", "FoV 140"]
byFOVTheta = [[],[]]
ThetaLabels = ["FoV 20", "FoV 140"]
for i,data in enumerate(fullAverageU):
    for x in range(0,len(data)):
        byFOVU[x].append(data[x])
for i,data in enumerate(fullAverageTheta):
    for x in range(0,len(data)):
        byFOVTheta[x].append(data[x])

print(f_oneway(byFOVU[0], byFOVU[1],byFOVU[2],byFOVU[3],byFOVU[4],byFOVU[5]))
print(f_oneway(byFOVTheta[0],byFOVTheta[1]))

plt.figure(figure,figsize=(10,6))
figure+=1
plt.boxplot(byFOVU,tick_labels=ULabels)
plt.title('Velocity Error Distribution Across Different FoVs')
plt.xlabel('Field of View (FoV)')
plt.ylabel('Velocity Error')
plt.savefig(writeFolder+"Boxplot_U")

plt.figure(figure,figsize=(10,6))
figure+=1
plt.boxplot(byFOVTheta,tick_labels=ThetaLabels)
plt.title('Pitch Angle Error Distribution Across Different FoVs')
plt.xlabel('Field of View (FoV)')
plt.ylabel('Pitch Error')
plt.savefig(writeFolder+"Boxplot_Theta")
plt.show()