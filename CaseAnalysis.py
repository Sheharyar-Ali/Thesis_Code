#%%
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
files = ["Data/Case_1.csv","Data/Case_2.csv","Data/Case_3.csv","Data/Case_4.csv","Data/Case_5.csv","Data/Case_6.csv"]
allBestPos=[]
figure =1
Case =1
allPD =pd.DataFrame({})
for filename in files:
    print(filename)
    fovRange = []
    horOpt =[]
    verOpt=[]
    maxVal=[]
    maxValt=[]
    maxValr =[]
    bestPos=[]
    file = open(filename)
    csvreader = csv.reader(file)
    for row in csvreader:
        if(row[0] !=""):
            fovRange.append(float(row[1]))
            horOpt.append(float(row[2]))
            verOpt.append(float(row[3]))
            maxVal.append(float(row[4]))
            maxValt.append(float(row[5]))
            maxValr.append(float(row[6]))
            bestPos.append(row[7])
    allBestPos.append(bestPos)

    plt.figure(figure)
    figure +=1
    plt.title("Best FoVs for Case "+ str(Case))
    plt.ylabel("Viewing angle for the optimal point [deg]")
    plt.xlabel("FoV tested [deg]")
    plt.plot(fovRange,np.array(horOpt),label="Horizontal viewing angle",marker="x")
    plt.plot(fovRange,np.array(verOpt),label="Vertical viewing angle",marker="x")
    plt.legend()
    plt.savefig("Images/fovC"+str(Case))

    plt.figure(figure)
    figure +=1
    plt.title("Max Delta for Case " + str(Case))
    plt.ylabel("Delta")
    plt.xlabel("FoV tested [deg]")
    plt.plot(fovRange,np.array(maxVal),marker="x",label="total")
    plt.plot(fovRange,np.array(maxValt),marker="x",label="translation")
    plt.plot(fovRange,np.array(maxValr),marker="x",label="rotation")
    plt.legend()
    plt.savefig("Images/deltaC"+str(Case))

    maxVal = np.array(maxVal)
    percentDiff = (maxVal[-1] - maxVal)/maxVal[-1] * 100
    allPD.insert(Case-1,"Case "+str(Case),percentDiff)


    plt.figure(figure)
    figure +=1
    plt.title("Percent Decrease for Case " + str(Case))
    plt.ylabel("Percent Decrease [%]")
    plt.xlabel("FoV tested [deg]")
    plt.plot(fovRange,np.array(percentDiff),marker="x",label="total")
    plt.legend()
    plt.savefig("Images/pdC"+str(Case))

    Case+=1
    file.close()
allPD.to_csv("Data/percentDiffs.csv")

# %%
