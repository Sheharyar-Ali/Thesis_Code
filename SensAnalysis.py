#%%
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
uRange = np.arange(1,7,1)
thetaRange= np.arange(-6,0,1)
uFiles = ["SensCase_1.csv","SensCase_4.csv"]
thetaFiles = ["SensCase_2.csv","SensCase_5.csv"]
bothFiles = ["SensCase_3.csv","SensCase_6.csv"]
files = ["SensCase_1.csv","SensCase_2.csv","SensCase_3.csv","SensCase_4.csv","SensCase_5.csv","SensCase_6.csv"]
figure = 1

fovRange = [20,40,60,80,100,120,140]
names =["Horizontal optimal","Vertical optimal","maxVal"]

Case1 = pd.read_csv("Data/SensCase_1.csv")
Case2 = pd.read_csv("Data/SensCase_2.csv")
Case3 = pd.read_csv("Data/SensCase_3.csv")
Case4 = pd.read_csv("Data/SensCase_4.csv")
Case5 = pd.read_csv("Data/SensCase_5.csv")
Case6 = pd.read_csv("Data/SensCase_6.csv")
#%%
# Case 1
count = 1
fig1,horAx=plt.subplots()
horAx.set_title("Optimal Horizontal Viewing Angle for Case 1")
horAx.set_ylabel("Viewing angle [deg]")
fig2,verAx=plt.subplots()
verAx.set_title("Optimal Vertical Viewing Angle for Case 1")
verAx.set_ylabel("Viewing angle [deg]")
fig3,maxAx=plt.subplots()
maxAx.set_title("Max Delta Values for Case 1")
maxAx.set_ylabel("Delta]")
for i in range(0,len(uRange)):
    u=uRange[i]
    horIndex= count*3 -1
    horName = Case1.columns[horIndex]
    verIndex= count*3 
    verName = Case1.columns[verIndex]
    maxIndex= count*3 + 1
    maxName = Case1.columns[maxIndex]
    Case1.plot(x="FoV",y=horName,ax=horAx,label="u = " + str(u) +" m/s",legend=True,xlabel="FoV [deg]")
    Case1.plot(x="FoV",y=verName,ax=verAx,label="u = " + str(u) +" m/s",legend=True,xlabel="FoV [deg]")
    Case1.plot(x="FoV",y=maxName,ax=maxAx,label="u = " + str(u) +" m/s",legend=True,xlabel="FoV [deg]")
    count+=1
fig1.savefig("Images/HorSensC1.png")
fig2.savefig("Images/VerSensC1.png")
fig3.savefig("Images/MaxSensC1.png")
#%%
# Case 2
count = 1
fig1,horAx=plt.subplots()
horAx.set_title("Optimal Horizontal Viewing Angle for Case 2")
horAx.set_ylabel("Viewing angle [deg]")
fig2,verAx=plt.subplots()
verAx.set_title("Optimal Vertical Viewing Angle for Case 2")
verAx.set_ylabel("Viewing angle [deg]")
fig3,maxAx=plt.subplots()
maxAx.set_title("Max Delta Values for Case 2")
maxAx.set_ylabel("Delta]")
for i in range(0,len(thetaRange)):
    u=thetaRange[i]
    horIndex= count*3 -1
    horName = Case2.columns[horIndex]
    verIndex= count*3 
    verName = Case2.columns[verIndex]
    maxIndex= count*3 + 1
    maxName = Case2.columns[maxIndex]
    Case2.plot(x="FoV",y=horName,ax=horAx,label="theta = " + str(u) +" deg",legend=True,xlabel="FoV [deg]")
    Case2.plot(x="FoV",y=verName,ax=verAx,label="theta = " + str(u) +" deg",legend=True,xlabel="FoV [deg]")
    Case2.plot(x="FoV",y=maxName,ax=maxAx,label="theta = " + str(u) +" deg",legend=True,xlabel="FoV [deg]")
    count+=1
fig1.savefig("Images/HorSensC2.png")
fig2.savefig("Images/VerSensC2.png")
fig3.savefig("Images/MaxSensC2.png")
#%%
# Case 3
count = 1
fig1,horAx=plt.subplots()
horAx.set_title("Optimal Horizontal Viewing Angle for Case 3")
horAx.set_ylabel("Viewing angle [deg]")
fig2,verAx=plt.subplots()
verAx.set_title("Optimal Vertical Viewing Angle for Case 3")
verAx.set_ylabel("Viewing angle [deg]")
fig3,maxAx=plt.subplots()
maxAx.set_title("Max Delta Values for Case 3")
maxAx.set_ylabel("Delta]")
for u in range(0,len(uRange)):
    u = uRange[i]
    for j in range(0,len(thetaRange)):
        theta=thetaRange[j]
        horIndex= count*3 -1
        horName = Case3.columns[horIndex]
        verIndex= count*3 
        verName = Case3.columns[verIndex]
        maxIndex= count*3 + 1
        maxName = Case3.columns[maxIndex]
        Case3.plot(x="FoV",y=horName,ax=horAx,label="u = " + str(u) +" m/s theta =" +str(theta)+ " deg",xlabel="FoV [deg]",legend=False)
        Case3.plot(x="FoV",y=verName,ax=verAx,label="u = " + str(u) +" m/s theta =" +str(theta)+ " deg",xlabel="FoV [deg]",legend=False)
        Case3.plot(x="FoV",y=maxName,ax=maxAx,label="u = " + str(u) +" m/s theta =" +str(theta)+ " deg",xlabel="FoV [deg]",legend=False)
        count+=1
fig1.savefig("Images/HorSensC3.png")
fig2.savefig("Images/VerSensC3.png")
fig3.savefig("Images/MaxSensC3.png")
#%%
# Case 4
count = 1
fig1,horAx=plt.subplots()
horAx.set_title("Optimal Horizontal Viewing Angle for Case 4")
horAx.set_ylabel("Viewing angle [deg]")
fig2,verAx=plt.subplots()
verAx.set_title("Optimal Vertical Viewing Angle for Case 4")
verAx.set_ylabel("Viewing angle [deg]")
fig3,maxAx=plt.subplots()
maxAx.set_title("Max Delta Values for Case 4")
maxAx.set_ylabel("Delta]")
for i in range(0,len(uRange)):
    u=uRange[i]
    horIndex= count*3 -1
    horName = Case4.columns[horIndex]
    verIndex= count*3 
    verName = Case4.columns[verIndex]
    maxIndex= count*3 + 1
    maxName = Case4.columns[maxIndex]
    Case4.plot(x="FoV",y=horName,ax=horAx,label="u = " + str(u) +" m/s",legend=True,xlabel="FoV [deg]")
    Case4.plot(x="FoV",y=verName,ax=verAx,label="u = " + str(u) +" m/s",legend=True,xlabel="FoV [deg]")
    Case4.plot(x="FoV",y=maxName,ax=maxAx,label="u = " + str(u) +" m/s",legend=True,xlabel="FoV [deg]")
    count+=1
fig1.savefig("Images/HorSensC4.png")
fig2.savefig("Images/VerSensC4.png")
fig3.savefig("Images/MaxSensC4.png")
#%%
# Case 5
count = 1
fig1,horAx=plt.subplots()
horAx.set_title("Optimal Horizontal Viewing Angle for Case 5")
horAx.set_ylabel("Viewing angle [deg]")
fig2,verAx=plt.subplots()
verAx.set_title("Optimal Vertical Viewing Angle for Case 5")
verAx.set_ylabel("Viewing angle [deg]")
fig3,maxAx=plt.subplots()
maxAx.set_title("Max Delta Values for Case 5")
maxAx.set_ylabel("Delta]")
for i in range(0,len(thetaRange)):
    u=thetaRange[i]
    horIndex= count*3 -1
    horName = Case5.columns[horIndex]
    verIndex= count*3 
    verName = Case5.columns[verIndex]
    maxIndex= count*3 + 1
    maxName = Case5.columns[maxIndex]
    Case5.plot(x="FoV",y=horName,ax=horAx,label="theta = " + str(u) +" deg",legend=True,xlabel="FoV [deg]")
    Case5.plot(x="FoV",y=verName,ax=verAx,label="theta = " + str(u) +" deg",legend=True,xlabel="FoV [deg]")
    Case5.plot(x="FoV",y=maxName,ax=maxAx,label="theta = " + str(u) +" deg",legend=True,xlabel="FoV [deg]")
    count+=1
fig1.savefig("Images/HorSensC5.png")
fig2.savefig("Images/VerSensC5.png")
fig3.savefig("Images/MaxSensC5.png")
#%%
# Case 6
count = 1
fig1,horAx=plt.subplots()
horAx.set_title("Optimal Horizontal Viewing Angle for Case 6")
horAx.set_ylabel("Viewing angle [deg]")
fig2,verAx=plt.subplots()
verAx.set_title("Optimal Vertical Viewing Angle for Case 6")
verAx.set_ylabel("Viewing angle [deg]")
fig3,maxAx=plt.subplots()
maxAx.set_title("Max Delta Values for Case 6")
maxAx.set_ylabel("Delta]")
for u in range(0,len(uRange)):
    u = uRange[i]
    for j in range(0,len(thetaRange)):
        theta=thetaRange[j]
        horIndex= count*3 -1
        horName = Case6.columns[horIndex]
        verIndex= count*3 
        verName = Case6.columns[verIndex]
        maxIndex= count*3 + 1
        maxName = Case6.columns[maxIndex]
        Case6.plot(x="FoV",y=horName,ax=horAx,label="u = " + str(u) +" m/s theta =" +str(theta)+ " deg",xlabel="FoV [deg]",legend=False)
        Case6.plot(x="FoV",y=verName,ax=verAx,label="u = " + str(u) +" m/s theta =" +str(theta)+ " deg",xlabel="FoV [deg]",legend=False)
        Case6.plot(x="FoV",y=maxName,ax=maxAx,label="u = " + str(u) +" m/s theta =" +str(theta)+ " deg",xlabel="FoV [deg]",legend=False)
        count+=1
fig1.savefig("Images/HorSensC6.png")
fig2.savefig("Images/VerSensC6.png")
fig3.savefig("Images/MaxSensC6.png")
# %%
