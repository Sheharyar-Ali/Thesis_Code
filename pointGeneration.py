import numpy as np
import itertools
import pandas as pd
from scipy.signal import TransferFunction, lsim
from scipy.integrate import cumulative_trapezoid

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
    yCoords = np.arange(0,yCoordsMax,step-2)
    zCoords = np.array([0.2])
    Coords = [xCoords,yCoords,zCoords]
    allCoords = list(itertools.product(*Coords))
    
    filteredCoords = []
    xVals,yVals = [],[]
    for pos in allCoords:
        horCheck = np.degrees(np.abs(np.arctan(pos[1]/pos[0])))
        verCheck = np.degrees(np.abs(np.arctan(pos[2]/pos[0])))
        # if(horCheck <= fovHor/2 and verCheck<=fovVer/2):
        #     filteredCoords.append(pos)
        #     xVals.append(pos[0])
        #     yVals.append(pos[1])
        filteredCoords.append(pos)
        xVals.append(pos[0])
        yVals.append(pos[1])
    return np.array(filteredCoords),np.array(xVals),np.array(yVals)

def movement(pointOrigin,dx,theta):
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])    
    newOrigin = (Ry @ pointOrigin) - dx
    return newOrigin,newOrigin[0],newOrigin[1],newOrigin[2]

def RealisticMovement(input, MTheta1s,MQ,XTheta1s,g,XU,t):
    tf_theta1s_theta = TransferFunction([MTheta1s],[1,0])
    ss_tf_theta1s_theta = MTheta1s/-MQ
    num_hc2 = (XTheta1s/ ss_tf_theta1s_theta) - g
    tf_theta_u =  TransferFunction([num_hc2],[1,-XU])
    
    _,theta,_ = lsim(tf_theta1s_theta,U=input,T=t)
    _,u,_ = lsim(tf_theta_u,U=theta,T=t)
    xPos = cumulative_trapezoid(u,t,initial=0)
    output = pd.DataFrame({})
    output.insert(0,"Time", t)
    output.insert(1,"Input",input)
    output.insert(2,"Theta",theta)
    output.insert(3,"U",u)
    output.insert(4,"X",xPos)

    return output

t = np.linspace(0,4,600)
input = -1/100 * np.sin((1/2 * 3)*np.pi*t)
MTheta1s = 26.4
MQ = -1.8954
MU = 0.05
XTheta1s = -9.280
XU = -0.02
g = 9.80665
XQ = 0.6674
behaviour = RealisticMovement(input,MTheta1s,MQ,XTheta1s,g,XU,t)
behaviour.insert(3,"Theta_deg",np.degrees(behaviour["Theta"]))
xRange = [4,40]
step = 8
coords,oldX,oldY = generateCoords(140,100,xRange[0],xRange[1],step)
dfCut =[]
timestamps = [0.5,1,1.5,2,2.5,3,3.5,4]
tolerance = 0.005
folder = "Heli_Sim/Assets/Resources/"

print(coords)
coordsOrg = pd.DataFrame({"X":oldY, "Y": np.ones_like(oldY)*0.2 , "Z": oldX})
coordsOrg.to_csv(folder+"coordsOrg.csv")
behaviour.to_csv(folder+"exampleMove.csv")
for T in timestamps:
    toAdd = behaviour[(behaviour["Time"]>= T - tolerance ) & (behaviour["Time"] < T+tolerance) ]
    dfCut.append(toAdd.iloc[0])
for i in range(1,len(behaviour)):
    slice = behaviour.iloc[i]
    # print(slice)
    dx = np.array([slice["U"] * (slice["Time"] - behaviour.iloc[i-1]["Time"] ),0,0])
    newCoords =[]
    xChange=[]
    yChange=[]
    zChange=[]
    for point in coords:
        newPoint,newX,newY,newZ = movement(point,dx,slice["Theta"])
        newCoords.append(newPoint)
        xChange.append(newX)
        yChange.append(newY)
        zChange.append(newZ)
    newCoords = np.array(newCoords)
    xChange = np.array(xChange)
    yChange=np.array(yChange)
    zChange=np.array(zChange)
    # print(xChange)
    # print(newCoords)
    # exit()
    # name = "coordsNew"+ str(timestamps[i])+ ".csv"
    
    # coordsNew = pd.DataFrame({"X": yChange, "Y":-zChange, "Z": xChange})
    # coordsNew.to_csv(folder+name )
# print(coordsOrg)
# print(coordsNew)
