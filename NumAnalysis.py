#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import control as ctrl
from scipy.signal import TransferFunction, lsim
import matplotlib.animation as ani
from IPython import display
import sympy as sy
import itertools
import functools

def create_ground_points(x_range, y_range, z_range, num_points):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    z = np.linspace(z_range[0], z_range[1], num_points)
    xx, yy,zz = np.meshgrid(x, y,z)
    points_world = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return points_world
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
    zCoords = np.array([5])
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
def Spherical(x,y,z):
    azimuth = np.arctan(y/x)
    mod = np.sqrt(x**2 + y**2 + z**2)
    elevation = np.arccos(z/mod)
    return np.array([np.degrees(azimuth),np.degrees(elevation)])


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
def FoVMaxComparison(dt,xVals,yVals,behaviour,t):
    bestDAzDU, bestDAzDTheta, bestDEleDU, bestDEleDTheta =  np.zeros_like(t),np.zeros_like(t),np.zeros_like(t),np.zeros_like(t)
    bestDAzDU_angle, bestDAzDTheta_angle, bestDEleDU_angle, bestDEleDTheta_angle = 0,0,0,0
    for i,xVal in enumerate(xVals):
        dAzdU, dAzdTheta, dEledU,dEledTheta = Analytical(dt,-5,xVal,yVals[i],behaviour["Theta"],behaviour["U"])
        viewingAngle = np.round(np.degrees(np.arctan(yVals[i]/xVal)))
        if np.mean(dAzdU) > np.mean(bestDAzDU):
            bestDAzDU = dAzdU
            bestDAzDU_angle = viewingAngle
        if np.mean(dAzdTheta) > np.mean(bestDAzDTheta):
            bestDAzDTheta = dAzdTheta
            bestDAzDTheta_angle = viewingAngle
        if np.mean(dEledU) > np.mean(bestDEleDU):
            bestDEleDU = dEledU
            bestDEleDU_angle = viewingAngle
        if np.mean(dEledTheta) > np.mean(bestDEleDTheta):
            bestDEleDTheta = dEledTheta
            bestDEleDTheta_angle = viewingAngle                    
    return bestDAzDU, bestDAzDTheta, bestDEleDU, bestDEleDTheta,bestDAzDU_angle, bestDAzDTheta_angle, bestDEleDU_angle, bestDEleDTheta_angle
def FovComparison(dt,xVals,yVals,behaviour,nPoints,t):
    baseDAzDU, baseDAZDTheta, baseDEleDU, baseDEleDTheta =  np.zeros_like(t),np.zeros_like(t),np.zeros_like(t),np.zeros_like(t)
    for i,xVal in enumerate(xVals):
        dAzdU, dAzdTheta, dEledU,dEledTheta = Analytical(dt,-5,xVal,yVals[i],behaviour["Theta"],behaviour["U"])
        baseDAzDU += dAzdU
        baseDAZDTheta += dAzdTheta
        baseDEleDU += dEledU
        baseDEleDTheta += dEledTheta

    meanDAzDU = baseDAzDU/ nPoints
    meanDAzDTheta = baseDAZDTheta / nPoints
    meanDEleDU = baseDEleDU /nPoints
    meanDEleDTheta = baseDEleDTheta/nPoints
    return meanDAzDU,meanDAzDTheta,meanDEleDU,meanDEleDTheta

def Projection(observerOrigin,pointOrigin,radius):
    P = pointOrigin - observerOrigin
    modP = np.sqrt(P[0] **2 + P[1]**2 + P[2]**2)
    Q = (radius/modP) * P
    R = Q + observerOrigin
    spherical = Spherical(R[0],R[1],R[2])
    return R, spherical

def ProjRetina(observerOrigin,pointOrigin,radius,u,dt,theta):
    # print("Point origin: ", pointOrigin)

    
    R,oldSpherical = Projection(observerOrigin,pointOrigin,radius)
    
    # print("Spherical original: ", R)
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    rotatedR = Ry @ R
    newSphericalRotation = Spherical(rotatedR[0],rotatedR[1],rotatedR[2])
    
    
    dx = np.array([u * dt,0,0])
    translatedR, newSphericalTranslation = Projection(observerOrigin,pointOrigin-dx,radius)

    newOrigin = (Ry @ pointOrigin) - dx
    fullR,newSpherical = Projection(observerOrigin,newOrigin,radius)

    
    # print("Old azimuth elevation: ", oldSpherical)
    # print("New Azimuth elevation ", newSphericalRotation,newSphericalTranslation,newSpherical)
    # print("Spherical new: ", rotatedR, translatedR, fullR)

    return oldSpherical,newSpherical,R,rotatedR,translatedR,fullR

def RealisticMovement(input, MTheta1s,MQ,XTheta1s,g,XU,t):
    tf_theta1s_theta = TransferFunction([MTheta1s],[1,0])
    ss_tf_theta1s_theta = MTheta1s/-MQ
    num_hc2 = (XTheta1s/ ss_tf_theta1s_theta) - g
    tf_theta_u =  TransferFunction([num_hc2],[1,-XU])
    
    _,theta,_ = lsim(tf_theta1s_theta,U=input,T=t)
    _,u,_ = lsim(tf_theta_u,U=theta,T=t)
    output = pd.DataFrame({})
    output.insert(0,"Time", t)
    output.insert(1,"Input",input)
    output.insert(2,"Theta",theta)
    output.insert(3,"U",u)

    return output




## Setup
r=0.017
MTheta1s = 26.4
MQ = -1.8954
MU = 0.05
XTheta1s = -9.280
XU = -0.02
g = 9.80665
XQ = 0.6674
observerOrigin = np.array([0,0,0])
xRange = (3, 10)  # X range on the ground
yRange = (-27, 27)  # Y range on the ground
zHeight = (-5,-5)  # Z height for ground plane
nPoints = 20 # Number of points in each dimension
points_world = create_ground_points(xRange, yRange, zHeight, nPoints)



## Create input
t = np.linspace(0,4,600)
t = np.arange(0,4.5,0.1)
dt = t[1] - t[0]
input = -1/100 * np.sin((1/2 * 3)*np.pi*t)
# input = -1/100 * np.sin((1/2 * 10)*np.pi*t)
# input = -1/100 * np.ones_like(t)
# input[t<0.5] = 0
behaviour = RealisticMovement(input,MTheta1s,MQ,XTheta1s,g,XU,t)
behaviour.insert(3,"Theta_deg",np.degrees(behaviour["Theta"]))
figure =1 
plt.figure(figure)
figure+=1
ax=plt.gca()
behaviour.plot(kind="line", x="Time", y="Input", ax=ax)
plt.title("Variation of Control Input over time")
plt.xlabel('Time [s]')
plt.ylabel('Control Input [rad s]')
ax.yaxis.set_major_formatter( ticker.FormatStrFormatter('%.3f'))
plt.savefig("Visuals/Input")
plt.legend()

plt.figure(figure)
figure+=1
ax=plt.gca()
behaviour.plot(kind="line", x="Time", y="Theta_deg", ax=ax)
plt.title("Variation of Pitch Angle over time")
plt.xlabel('Time [s]')
plt.ylabel('Theta [deg]')
plt.savefig("Visuals/Theta")
plt.legend()

plt.figure(figure)
figure+=1
ax=plt.gca()
behaviour.plot(kind="line", x="Time", y="U", ax=ax)
plt.title("Variation of Forward Velocity over time")
plt.xlabel('Time [s]')
plt.ylabel('U [m/s]')
plt.savefig("Visuals/U")
plt.legend()
plt.close('all')

## Analytical ##
# Testing
testX = 2
testY = 2
testTheta = np.radians(15)
testU = 2
dtTest =0.1
dAzdU, dAzdTheta, dEledU,dEledTheta = Analytical(deltaT=dt,z=-5,xVal=testX,yVal=testY,thetaVals=testTheta,uVals=testU)


## Set up figures for analysing one Fov
fig_dAzDU = plt.figure(figure)
figure+=1
ax_dAZDU = fig_dAzDU.add_subplot(111)
ax_dAZDU.set_title("Variation of dAzimuth_d(Udt) wrt x and y for Fov = 140 deg")
ax_dAZDU.set_xlabel("Time [s]")
ax_dAZDU.set_ylabel("dAzimuth_d(Udt) [rad / m]")


fig_dAzDTheta = plt.figure(figure)
figure+=1
ax_dAZDTheta = fig_dAzDTheta.add_subplot(111)
ax_dAZDTheta.set_title("Variation of dAzimuth_dTheta wrt x and y for Fov = 140 deg")
ax_dAZDTheta.set_xlabel("Time [s]")
ax_dAZDTheta.set_ylabel("dAzimuth_dTheta [-]")


fig_dEleDU = plt.figure(figure)
figure+=1
ax_dEleDU = fig_dEleDU.add_subplot(111)
ax_dEleDU.set_title("Variation of dElevation_d(Udt) wrt x and y for Fov = 140 deg")
ax_dEleDU.set_xlabel("Time [s]")
ax_dEleDU.set_ylabel("dElevation_d(Udt) [rad / m]")


fig_dEleDTheta = plt.figure(figure)
figure+=1
ax_dEleDTheta = fig_dEleDTheta.add_subplot(111)
ax_dEleDTheta.set_title("Variation of dElevation_dTheta wrt x and y for Fov = 140 deg")
ax_dEleDTheta.set_xlabel("Time [s]")
ax_dEleDTheta.set_ylabel("dElevation_dTheta [-]")


# Analyse one FoV
xVals = np.arange(xRange[0],xRange[1]+1,1)
yVals = xVals * np.tan(np.radians(140/2))
for i,xPos in enumerate(xVals):
    dAzdU, dAzdTheta, dEledU,dEledTheta = Analytical(deltaT=dt,z=-5,xVal=xPos,yVal=yVals[i],thetaVals=behaviour["Theta"],uVals=behaviour["U"])
    yVals[i] = round(yVals[i],2)
    ax_dAZDU.plot(t,dAzdU,label="x=" + str(xPos) + " y=" + str(yVals[i]))
    ax_dAZDTheta.plot(t,dAzdTheta,label="x=" + str(xPos) + " y=" + str(yVals[i]))
    ax_dEleDU.plot(t,dEledU,label="x=" + str(xPos) + " y=" + str(yVals[i]))
    ax_dEleDTheta.plot(t,dEledTheta,label="x=" + str(xPos) + " y=" + str(yVals[i])) 
    
ax_dAZDU.legend()    
ax_dAZDTheta.legend()   
ax_dEleDU.legend()
ax_dEleDTheta.legend()
fig_dAzDU.savefig("Visuals/dAzDUdt_fov140.png")
fig_dAzDTheta.savefig("Visuals/dAzDTheta_fov140.png")
fig_dEleDTheta.savefig("Visuals/dEleDTheta_fov140.png")
fig_dEleDU.savefig("Visuals/dEleDUdt_fov140.png")    
# plt.show()
# exit()
plt.close('all')


## Set up figures for FoV Comparison
fig_dAzDU = plt.figure(figure)
figure+=1
ax_dAZDU = fig_dAzDU.add_subplot(111)
ax_dAZDU.set_title("Variation of dAzimuth_d(Udt) wrt FoV")
ax_dAZDU.set_xlabel("Time [s]")
ax_dAZDU.set_ylabel("dAzimuth_d(Udt) [rad / m]")


fig_dAzDTheta = plt.figure(figure)
figure+=1
ax_dAZDTheta = fig_dAzDTheta.add_subplot(111)
ax_dAZDTheta.set_title("Variation of dAzimuth_dTheta wrt FoV")
ax_dAZDTheta.set_xlabel("Time [s]")
ax_dAZDTheta.set_ylabel("dAzimuth_dTheta [-]")


fig_dEleDU = plt.figure(figure)
figure+=1
ax_dEleDU = fig_dEleDU.add_subplot(111)
ax_dEleDU.set_title("Variation of dElevation_d(Udt) wrt FoV")
ax_dEleDU.set_xlabel("Time [s]")
ax_dEleDU.set_ylabel("dElevation_d(Udt) [rad / m]")


fig_dEleDTheta = plt.figure(figure)
figure+=1
ax_dEleDTheta = fig_dEleDTheta.add_subplot(111)
ax_dEleDTheta.set_title("Variation of dElevation_dTheta wrt FoV")
ax_dEleDTheta.set_xlabel("Time [s]")
ax_dEleDTheta.set_ylabel("dElevation_dTheta [-]")


# Analyse all FoVs
baseCase = np.array([1,1,-5])
xVals = np.linspace(xRange[0], xRange[1], nPoints)
FoVRange = np.arange(10,160,10)
allDAzDu = []
allDAzDTheta = []
allDEleDU = []
allDEleDTheta = []
colors = [
    (255, 0, 0),       # Red
    (0, 255, 0),       # Green
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (0, 255, 255),     # Cyan
    (255, 0, 255),     # Magenta
    (255, 165, 0),     # Orange
    (128, 0, 128),     # Purple
    (255, 192, 203),   # Pink
    (165, 42, 42),     # Brown
    (128, 128, 128),   # Gray
    (0, 0, 0),         # Black
    (128, 255, 0),     # Lime
    (0, 0, 128),       # Navy
    (128, 128, 0)      # Olive
]

colors= [(r/255, g/255, b/255) for r, g, b in colors]
for i,angle in enumerate(FoVRange):
    yVals = xVals * np.tan(np.radians(angle/2))
    meanDAzDU,meanDAzDTheta,meanDEleDU,meanDEleDTheta = FovComparison(dt,xVals,yVals,behaviour,nPoints,t)
    allDAzDu.append(meanDAzDU)
    allDAzDTheta.append(meanDAzDTheta)
    allDEleDU.append(meanDEleDU)
    allDEleDTheta.append(meanDEleDTheta)
    ax_dAZDU.plot(t,meanDAzDU,label=str(angle) + " deg",color=colors[i])
    ax_dAZDTheta.plot(t,meanDAzDTheta,label=str(angle) + " deg",color=colors[i])
    ax_dEleDU.plot(t,meanDEleDU,label=str(angle) + " deg",color=colors[i])
    ax_dEleDTheta.plot(t,meanDEleDTheta,label=str(angle) + " deg",color=colors[i]) 

ax_dAZDU.legend()    
ax_dAZDTheta.legend()   
ax_dEleDU.legend()
ax_dEleDTheta.legend()
fig_dAzDU.savefig("Visuals/dAzDUdt.png")
fig_dAzDTheta.savefig("Visuals/dAzDTheta.png")
fig_dEleDTheta.savefig("Visuals/dEleDTheta.png")
fig_dEleDU.savefig("Visuals/dEleDUdt.png")

allMeans_dAzDU=[0]
allMeans_dAzDTheta=[0]
allMeans_dEleDU=[0]
allMeans_dEleDTheta=[0]
plt.figure(figure)
figure+=1
plt.title("Difference between mean values for FoVs")
plt.ylabel("Difference")
plt.xlabel("FoV (deg)")
for i in range(1,len(allDAzDu)):
    m = allDAzDu[i] - allDAzDu[0]
    allMeans_dAzDU.append(np.mean(m))
    m = allDAzDTheta[i] - allDAzDTheta[0]
    allMeans_dAzDTheta.append(np.mean(m)) 
    m = allDEleDU[i] - allDEleDU[0]
    allMeans_dEleDU.append(np.mean(m))   
    m = allDEleDTheta[i] - allDEleDTheta[0]
    allMeans_dEleDTheta.append(np.mean(m))   
plt.plot(FoVRange,allMeans_dAzDU,label="dAz_d(Udt)",marker = "x")
plt.plot(FoVRange,allMeans_dAzDTheta,label="dAz_dTheta",marker = "x")
plt.plot(FoVRange,allMeans_dEleDU,label="dEle_d(Udt)",marker = "x")
plt.plot(FoVRange,allMeans_dEleDTheta,label="dEle_dTheta",marker = "x")
plt.legend()
plt.savefig("Visuals/FOVBehaviour_sine")
# plt.show()
plt.close('all')
for i,angle in enumerate(FoVRange):
    allCoords,allX,allY = generateCoords(angle,100,xRange[0],xRange[1],0.1)
    print("angle ", angle, " points ", len(allCoords))




# Setting up figures for max movement analysis
fig_dAzDU = plt.figure(figure)
figure+=1
ax_best_dAZDU = fig_dAzDU.add_subplot(111)
ax_best_dAZDU.set_title("Best dAzimuth_d(Udt) wrt FoV")
ax_best_dAZDU.set_xlabel("Time [s]")
ax_best_dAZDU.set_ylabel("dAzimuth_d(Udt) [rad s/ m]")


fig_dAzDTheta = plt.figure(figure)
figure+=1
ax_best_dAZDTheta = fig_dAzDTheta.add_subplot(111)
ax_best_dAZDTheta.set_title("Best dAzimuth_dTheta wrt FoV")
ax_best_dAZDTheta.set_xlabel("Time [s]")
ax_best_dAZDTheta.set_ylabel("dAzimuth_dTheta [-]")


fig_dEleDU = plt.figure(figure)
figure+=1
ax_best_dEleDU = fig_dEleDU.add_subplot(111)
ax_best_dEleDU.set_title("Best dElevation_d(Udt) wrt FoV")
ax_best_dEleDU.set_xlabel("Time [s]")
ax_best_dEleDU.set_ylabel("dElevation_dU [rad s/ m]")


fig_dEleDTheta = plt.figure(figure)
figure+=1
ax_best_dEleDTheta = fig_dEleDTheta.add_subplot(111)
ax_best_dEleDTheta.set_title("Best dElevation_dTheta wrt FoV")
ax_best_dEleDTheta.set_xlabel("Time [s]")
ax_best_dEleDTheta.set_ylabel("dElevation_dTheta [-]")

# Calculate max value
best_DAzDu = []
best_DAzDTheta = []
best_DEleDU = []
best_DEleDTheta = []
for i,angle in enumerate(FoVRange):
    print("analysing: ", angle)
    allCoords,allX,allY = generateCoords(angle,100,xRange[0],xRange[1],0.1)
    bestDAzDU, bestDAzDTheta, bestDEleDU, bestDEleDTheta,\
    bestDAzDU_angle, bestDAzDTheta_angle, bestDEleDU_angle, bestDEleDTheta_angle = FoVMaxComparison(dt,allX,allY,behaviour,t)
    best_DAzDu.append(bestDAzDU)
    best_DAzDTheta.append(bestDAzDTheta)
    best_DEleDU.append(bestDEleDU)
    best_DEleDTheta.append(bestDEleDTheta)
    ax_best_dAZDU.plot(t,bestDAzDU,label="FoV: "+str(angle) + ", Viewing Angle: "+ str(bestDAzDU_angle),color=colors[i])
    ax_best_dAZDTheta.plot(t,bestDAzDTheta,label="FoV: "+str(angle) + ", Viewing Angle: "+ str(bestDAzDTheta_angle),color=colors[i])
    ax_best_dEleDU.plot(t,bestDEleDU,label="FoV: "+str(angle) + ", Viewing Angle: "+ str(bestDEleDU_angle),color=colors[i])
    ax_best_dEleDTheta.plot(t,bestDEleDTheta,label="FoV: "+str(angle) + ", Viewing Angle: "+ str(bestDEleDTheta_angle),color=colors[i])     

ax_best_dAZDU.legend(bbox_to_anchor=(1.05, 1), loc='upper left')    
ax_best_dAZDTheta.legend(bbox_to_anchor=(1.05, 1), loc='upper left')   
ax_best_dEleDU.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_best_dEleDTheta.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig_dAzDU.tight_layout()
fig_dAzDTheta.tight_layout()
fig_dEleDU.tight_layout()
fig_dEleDTheta.tight_layout()
fig_dAzDU.savefig("Visuals/dAzDU_best.png")
fig_dAzDTheta.savefig("Visuals/dAzDTheta_best.png")
fig_dEleDTheta.savefig("Visuals/dEleDTheta_best.png")
fig_dEleDU.savefig("Visuals/dEleDU_best.png")

# plt.show() 
# exit()
plt.close('all')





    
    
    
    
    
    