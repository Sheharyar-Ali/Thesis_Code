import numpy as np
import itertools
import pandas as pd
from scipy.signal import TransferFunction, lsim
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

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
    yCoords = np.arange(-yCoordsMax,yCoordsMax,step)
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

# t = np.linspace(0,4,600)
# Create input
t = np.arange(0,4.5,0.1)
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
thetaDot = np.diff(a=behaviour['Theta_deg'],prepend=0) / np.diff(a=behaviour["Time"],prepend=0)
behaviour.insert(6,"Theta_dot_deg", thetaDot)
xRange = [2,30]
step = 3

# Generate list of pointsa to show in Unity
coords,oldX,oldY = generateCoords(140,100,xRange[0],xRange[1],step)
print(coords)

# Export data for Unity
folder = "Heli_Sim/Assets/Resources/"
coordsOrg = pd.DataFrame({"X":oldY, "Y": np.ones_like(oldY)*0.2 , "Z": oldX})
coordsOrg.to_csv(folder+"coordsOrg.csv")
behaviour.to_csv(folder+"exampleMove.csv")

