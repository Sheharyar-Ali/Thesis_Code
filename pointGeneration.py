import numpy as np
import itertools
import pandas as pd
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

xRange = [1,30]
step = 2
u = 0
theta = np.radians(-5)
dt = 0.1
folder = "Heli_Sim/Assets/Scripts/"
coords,oldX,oldY = generateCoords(140,100,xRange[0],xRange[1],step)
print(coords)
newCoords =[]
xChange=[]
yChange=[]
zChange=[]
dx = dx = np.array([u * dt,0,0])
for point in coords:
    newPoint,newX,newY,newZ = movement(point,dx,theta)
    newCoords.append(newPoint)
    xChange.append(newX)
    yChange.append(newY)
    zChange.append(newZ)
newCoords = np.array(newCoords)
xChange = np.array(xChange)
yChange=np.array(yChange)
zChange=np.array(zChange)
# print(xChange)
print(newCoords)

coordsOrg = pd.DataFrame({"X":oldY, "Y": np.ones_like(oldY)*0.2 , "Z": oldX})
coordsOrg.to_csv(folder+"coordsOrg.csv")
coordsNew = pd.DataFrame({"X": yChange, "Y":zChange, "Z": xChange})
coordsNew.to_csv(folder+"coordsNew.csv" )
# print(coordsOrg)
# print(coordsNew)
