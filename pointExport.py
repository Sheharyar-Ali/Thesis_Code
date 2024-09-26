
import numpy as np
import itertools

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

xRange = [1,10]
step = 0.5
coords,_,_ = generateCoords(140,100,xRange[0],xRange[1],step)

# Basic ones
pitch = -6
u = 0.2
newCoords = []
