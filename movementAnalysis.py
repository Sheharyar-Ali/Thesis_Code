import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sy
import itertools
from scipy.integrate import cumtrapz

def RMSE(data):
    error = np.zeros_like(data) - data
    rmse = np.sqrt(np.sum(error**2) / len(error))
    return rmse
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
    return Time,controlInput,controlVelocity,ffVelocity,heliVelocity

def generateCoords(fovHor,fovVer,start,end,step):
    """Generate coordiantes within a specific horizontal Fov

    Args:
        fovHor (float): Horizontal fov in deg
        fovVer (float): Vertical fov in deg
        start (float): starting x-coordinate
        end (float): ending x-coordinate
        step (float): step of x range and y range

    Returns:
        list of cooridnates, list of x-coordinates, list of y-coordinates
    """
    xCoords = np.arange(start,end,step)
    yCoordsMax = np.tan(np.radians(fovHor/2)) * end
    yCoords = np.arange(0,yCoordsMax,step)
    zCoords = np.array([-5])
    Coords = [xCoords,yCoords,zCoords]
    allCoords = list(itertools.product(*Coords))
    
    filteredCoords = []
    xVals,yVals = [],[]
    for pos in allCoords:
        horCheck = np.degrees(np.abs(np.arctan(pos[1]/pos[0])))
        verCheck = np.degrees(np.abs(np.arctan(pos[2]/pos[0])))
        if(horCheck <= fovHor/2 and verCheck<=fovVer/2):
            filteredCoords.append(pos)
            xVals.append(pos[0])
            yVals.append(pos[1])
    return filteredCoords,xVals,yVals

def Analytical(deltaT,z,xVal,yVal,thetaVals,uVals):
    x = sy.Symbol("x")
    y = sy.Symbol("y")
    # u = sy.Symbol("u")
    # dt = sy.Symbol("dt")
    dist = sy.Symbol("dist")
    theta = sy.Symbol("theta")
    Ry = np.array([[sy.cos(theta), 0, sy.sin(theta)],
                   [0, 1, 0],
                   [-sy.sin(theta), 0, sy.cos(theta)]])
    PNew = Ry @ np.array([x,y,z]) - [dist,0,0]
    Azimuth = sy.atan(PNew[1] / PNew[0])
    modulus = sy.sqrt(PNew[0] **2 + PNew[1] **2 + PNew[2] **2)
    Elevation = sy.acos(PNew[2] / modulus)
    dAz_dU = sy.lambdify((x,y,theta,dist),Azimuth.diff(dist))
    dEle_dU = sy.lambdify((x,y,theta,dist),Elevation.diff(dist))
    dAz_dTheta = sy.lambdify((x,y,theta,dist),Azimuth.diff(theta))
    dEle_dTheta = sy.lambdify((x,y,theta,dist),Elevation.diff(theta))


    Eval_dAzdU = dAz_dU(xVal,yVal,thetaVals,uVals*deltaT)
    Eval_dAzdTheta = dAz_dTheta(xVal,yVal,thetaVals,uVals*deltaT)
    Eval_dEledU = dEle_dU(xVal,yVal,thetaVals,uVals*deltaT)
    Eval_dEledTheta = dEle_dTheta(xVal,yVal,thetaVals,uVals*deltaT)
    return Eval_dAzdU, Eval_dAzdTheta, Eval_dEledU,Eval_dEledTheta



readFolder = "ExpData/ILAH/"
fileName = "export_ilah176,1395362,3279552,7027_actual_140.csv"
Time,controlInput,_,_,velocity = readFileU(readFolder+fileName)
np.array(controlInput)
figure = 1
dt = 0.01
MTheta1s = 26.4
finalAngle = controlInput * MTheta1s * dt
currentEuler = [finalAngle[0]]
for i in range(0,len(finalAngle)-1):
    currentEuler.append(currentEuler[i] + finalAngle[i])
    
print(len(currentEuler))


velFixed =[]
tFixed=[]
pitchFixed=[]
for i in range(1,len(Time)):
    if (Time[i] != Time[i-1]):
        velFixed.append(velocity[i])
        tFixed.append(Time[i])
        pitchFixed.append(currentEuler[i])
tFixed=np.array(tFixed)
velFixed=np.array(velFixed)
pitchFixed = np.array(pitchFixed)    
pitchRad = pitchFixed * (np.pi/180)

df = pd.DataFrame({"Time": tFixed, "V": velFixed , "Pitch_Deg": pitchFixed, "Pitch_Rad": pitchRad})
df.to_csv("Heli_Sim/Assets/Resources/actualMove.csv")


plt.figure(figure)
figure+=1
plt.title("Progression of Velocity")
plt.ylabel("Velocity [m/s]")
plt.xlabel("Time [s]")
plt.plot(Time,velocity)
plt.savefig("Visuals/VelocityExp")

plt.figure(figure)
figure+=1
plt.title("Progression of Pitch")
plt.ylabel("Pitch [deg]")
plt.xlabel("Time [s]")
plt.plot(Time,currentEuler,label="Actual")
# plt.plot(Time,Az_t * (180/np.pi),label="new")
plt.savefig("Visuals/ThetaExp")


# Over all points
xRange=[1,8]
step = 0.2
coords,_,_ = generateCoords(140,100,xRange[0],xRange[1],step)
alldAzDu,alldAzdTheta, alldEledU,alldEledTheta = [],[],[],[]
for pos in coords:
    dAzdU, dAzdTheta, dEledU,dEledTheta = Analytical(tFixed[1] - tFixed[0], pos[2],pos[0],pos[1],pitchRad,velFixed) 
    alldAzDu.append(dAzdU)
    alldAzdTheta.append(dAzdTheta)
    alldEledU.append(dEledU )
    alldEledTheta.append(dEledTheta)

meanDAzDU = np.mean(alldAzDu,axis=0)
meanDAzDTheta = np.mean(alldAzdTheta,axis=0)
meanDEleDU = np.mean(alldEledU,axis=0)
meanDEleDTheta = np.mean(alldEledTheta,axis=0)
stdDAzDu = RMSE(meanDAzDU)
stdDAzDTheta = RMSE(meanDAzDTheta)
stdDEleDu = RMSE(meanDEleDU)
stdDEleDTheta = RMSE(meanDEleDTheta)

Az_t_theta = cumtrapz(meanDAzDTheta,pitchFixed,initial=0)
Az_t_U = cumtrapz(meanDAzDU,(velFixed * (tFixed[1] - tFixed[0])),initial=0)

print(alldAzDu)
print(stdDAzDu,stdDAzDTheta)
print(stdDEleDu,stdDEleDTheta)

# plt.figure(figure)
# figure+=1
# plt.plot(Time,udt)
# plt.show()
plt.figure(figure)
figure+=1
plt.plot(tFixed,Az_t_U* (180/np.pi))
plt.figure(figure)
figure+=1
plt.plot(tFixed,Az_t_theta* (180/np.pi))

plt.figure(figure)
figure+=1
plt.title("Average variation of dAzimuth with respect to (Udt) and Theta over Time")
plt.xlabel("Time [s]")
plt.ylabel("dAzimuth_d [rad/m or -]")
plt.plot(tFixed,meanDAzDU,label="dAzimuth_d(Udt)")
plt.plot(tFixed,meanDAzDTheta,label="dAzimuth_dTheta")
plt.legend()
plt.savefig("Visuals/dAzimuth_Exp")

plt.figure(figure)
figure+=1
plt.title("Average Variation of dElevation with respect to (Udt) and Theta over Time")
plt.xlabel("Time [s]")
plt.ylabel("dElevation_d [rad/m or -]")
plt.plot(tFixed,meanDEleDU,label="dElevation_d(Udt)")
plt.plot(tFixed,meanDEleDTheta,label="dElevation_dTheta")
plt.legend()
plt.savefig("Visuals/dElevation_Exp")

# Over points at 90 deg
print("90 Deg now")
plt.close('all')
fig_dAzDU = plt.figure(figure)
figure+=1
ax_dAZDU = fig_dAzDU.add_subplot(111)
ax_dAZDU.set_title("Variation of dAzimuth_d(Udt) wrt the two views")
ax_dAZDU.set_xlabel("Time [s]")
ax_dAZDU.set_ylabel("dAzimuth_d(Udt) [rad / m]")


fig_dAzDTheta = plt.figure(figure)
figure+=1
ax_dAZDTheta = fig_dAzDTheta.add_subplot(111)
ax_dAZDTheta.set_title("Variation of dAzimuth_dTheta wrt the two views")
ax_dAZDTheta.set_xlabel("Time [s]")
ax_dAZDTheta.set_ylabel("dAzimuth_dTheta [-]")


fig_dEleDU = plt.figure(figure)
figure+=1
ax_dEleDU = fig_dEleDU.add_subplot(111)
ax_dEleDU.set_title("Variation of dElevation_d(Udt) wrt the two views")
ax_dEleDU.set_xlabel("Time [s]")
ax_dEleDU.set_ylabel("dElevation_d(Udt) [rad / m]")


fig_dEleDTheta = plt.figure(figure)
figure+=1
ax_dEleDTheta = fig_dEleDTheta.add_subplot(111)
ax_dEleDTheta.set_title("Variation of dElevation_dTheta wrt the two views")
ax_dEleDTheta.set_xlabel("Time [s]")
ax_dEleDTheta.set_ylabel("dElevation_dTheta [-]")


xVals = np.arange(xRange[0],xRange[1]+1,1)
yVals = xVals * np.tan(np.radians(90/2))
alldAzDu,alldAzdTheta, alldEledU,alldEledTheta = [],[],[],[]
for i,xPos in enumerate(xVals):
    dAzdU, dAzdTheta, dEledU,dEledTheta = Analytical(tFixed[1] - tFixed[0], -5,xPos,yVals[i],pitchRad,velFixed) 
    alldAzDu.append(dAzdU)
    alldAzdTheta.append(dAzdTheta)
    alldEledU.append(dEledU )
    alldEledTheta.append(dEledTheta)

meanDAzDU = np.mean(alldAzDu,axis=0)
meanDAzDTheta = np.mean(alldAzdTheta,axis=0)
meanDEleDU = np.mean(alldEledU,axis=0)
meanDEleDTheta = np.mean(alldEledTheta,axis=0)
stdDAzDu = RMSE(meanDAzDU)
stdDAzDTheta = RMSE(meanDAzDTheta)
stdDEleDu = RMSE(meanDEleDU)
stdDEleDTheta = RMSE(meanDEleDTheta)

Az_t_theta = cumtrapz(meanDAzDTheta,pitchFixed,initial=0)
Az_t_U = cumtrapz(meanDAzDU,(velFixed * (tFixed[1] - tFixed[0])),initial=0)

#print(alldAzDu)
print("RMSE")
print(stdDAzDu,stdDAzDTheta)
print(stdDEleDu,stdDEleDTheta)
print("Means")
print(np.mean(meanDAzDU), np.mean(meanDAzDTheta))
print(np.mean(meanDEleDU),np.mean(meanDEleDTheta))
ax_dAZDU.plot(tFixed,meanDAzDU,label="45 deg")
ax_dAZDTheta.plot(tFixed,meanDAzDTheta,label="45 deg")
ax_dEleDU.plot(tFixed,meanDEleDU,label="45 deg")
ax_dEleDTheta.plot(tFixed,meanDEleDTheta,label="45 deg")
plt.figure(figure)
figure+=1
plt.title("Variation of dAzimuth with respect to (Udt)\n and Theta over Time for points at +45 degrees")
plt.xlabel("Time [s]")
plt.ylabel("dAzimuth_d [rad/m or -]")
plt.plot(tFixed,meanDAzDU,label="dAzimuth_d(Udt)")
plt.plot(tFixed,meanDAzDTheta,label="dAzimuth_dTheta")
plt.legend()
plt.savefig("Visuals/dAzimuth_90")

plt.figure(figure)
figure+=1
plt.title("Average Variation of dElevation with respect to (Udt)\n and Theta over Time for points at +45degrees")
plt.xlabel("Time [s]")
plt.ylabel("dElevation_d [rad/m or -]")
plt.plot(tFixed,meanDEleDU,label="dElevation_d(Udt)")
plt.plot(tFixed,meanDEleDTheta,label="dElevation_dTheta")
plt.legend()
plt.savefig("Visuals/dElevation_90")    


# Over points at 15 deg
print("15 Deg now")
xVals = np.arange(xRange[0],xRange[1]+1,1)
yVals = xVals * np.tan(np.radians(15/2))
alldAzDu,alldAzdTheta, alldEledU,alldEledTheta = [],[],[],[]
for i,xPos in enumerate(xVals):
    dAzdU, dAzdTheta, dEledU,dEledTheta = Analytical(tFixed[1] - tFixed[0], -5,xPos,yVals[i],pitchRad,velFixed) 
    alldAzDu.append(dAzdU)
    alldAzdTheta.append(dAzdTheta)
    alldEledU.append(dEledU )
    alldEledTheta.append(dEledTheta)

meanDAzDU = np.mean(alldAzDu,axis=0)
meanDAzDTheta = np.mean(alldAzdTheta,axis=0)
meanDEleDU = np.mean(alldEledU,axis=0)
meanDEleDTheta = np.mean(alldEledTheta,axis=0)
stdDAzDu = RMSE(meanDAzDU)
stdDAzDTheta = RMSE(meanDAzDTheta)
stdDEleDu = RMSE(meanDEleDU)
stdDEleDTheta = RMSE(meanDEleDTheta)

Az_t_theta = cumtrapz(meanDAzDTheta,pitchFixed,initial=0)
Az_t_U = cumtrapz(meanDAzDU,(velFixed * (tFixed[1] - tFixed[0])),initial=0)

#print(alldAzDu)
print("RMSE")
print(stdDAzDu,stdDAzDTheta)
print(stdDEleDu,stdDEleDTheta)
print("Means")
print(np.mean(meanDAzDU), np.mean(meanDAzDTheta))
print(np.mean(meanDEleDU),np.mean(meanDEleDTheta))
ax_dAZDU.plot(tFixed,meanDAzDU,label="7.5 deg")
ax_dAZDTheta.plot(tFixed,meanDAzDTheta,label="7.5 deg")
ax_dEleDU.plot(tFixed,meanDEleDU,label="7.5 deg")
ax_dEleDTheta.plot(tFixed,meanDEleDTheta,label="7.5 deg")

ax_dAZDU.legend()    
ax_dAZDTheta.legend()   
ax_dEleDU.legend()
ax_dEleDTheta.legend()
fig_dAzDU.savefig("Visuals/dAzDUdt_comparison.png")
fig_dAzDTheta.savefig("Visuals/dAzDTheta_comparison.png")
fig_dEleDTheta.savefig("Visuals/dEleDTheta_comparison.png")
fig_dEleDU.savefig("Visuals/dEleDUdt_comparison.png")    

plt.figure(figure)
figure+=1
plt.title("Variation of dAzimuth with respect to (Udt) \n and Theta over Time for points at +7.5 degrees")
plt.xlabel("Time [s]")
plt.ylabel("dAzimuth_d [rad/m or -]")
plt.plot(tFixed,meanDAzDU,label="dAzimuth_d(Udt)")
plt.plot(tFixed,meanDAzDTheta,label="dAzimuth_dTheta")
plt.legend()
plt.savefig("Visuals/dAzimuth_15")

plt.figure(figure)
figure+=1
plt.title("Average Variation of dElevation with respect to (Udt) \n and Theta over Time for points at +7.5 degrees")
plt.xlabel("Time [s]")
plt.ylabel("dElevation_d [rad/m or -]")
plt.plot(tFixed,meanDEleDU,label="dElevation_d(Udt)")
plt.plot(tFixed,meanDEleDTheta,label="dElevation_dTheta")
plt.legend()
plt.savefig("Visuals/dElevation_15")
plt.show()
# # angleWanted = pushValue * maxPitchRate / maxVal;
# # var thetaDot = angleWanted * M_theta1s;
# # finalAngle = thetaDot * dtPython;