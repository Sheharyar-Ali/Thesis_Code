# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from ForcingFunction import  ff,time,ff_dot
import itertools

def position_viewing(theta,phi,psi,pos_world,pos_COP):
    Rx = np.array([[1,0,0],[0,np.cos(phi),np.sin(phi)],[0,-np.sin(phi),np.cos(phi)]])
    Ry = np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
    Rz = np.array([[np.cos(psi),np.sin(psi),0],[-np.sin(psi),np.cos(psi),0],[0,0,1]])
    T = Rx @ Ry @ Rz
    pos_viewing = T @ (np.array(pos_world) - np.array(pos_COP))
    return pos_viewing

def kappa(screen_size,FoV):
    k = screen_size / (2* np.tan(np.radians(FoV)/2))
    return k
def up (kappa,ypv,xpv):
    up = kappa * (ypv/xpv)
    return up
def vp (kappa,zpv,xpv):
    vp = -kappa * (zpv/xpv)
    return vp
def up_dot(kappa,ub,vb,xpb,pb,qb,rb,up,vp):
    exp_1 = (-kappa * vb + up * ub) / xpb
    exp_2 = (-kappa**2 * rb - kappa*vp*pb - up*vp*qb - up**2 * rb) / kappa
    up_dot = exp_1 + exp_2
    return up_dot
def up_dot_comp(kappa,ub,vb,xpb,pb,qb,rb,up,vp):
    exp_1 = (-kappa * vb + up * ub) / xpb
    exp_2 = (-kappa**2 * rb - kappa*vp*pb - up*vp*qb - up**2 * rb) / kappa
    return exp_1,exp_2
def vp_dot(kappa,ub,wb,vp,xpb,pb,qb,rb,up):
    exp_3 = (kappa * wb + vp * ub) / xpb
    exp_4 = (-kappa **2 * qb + kappa * up*pb - up*vp*rb - vp**2 * qb) / kappa
    vp_dot = exp_3 + exp_4
    return vp_dot
def vp_dot_comp(kappa,ub,wb,vp,xpb,pb,qb,rb,up):
    exp_3 = (kappa * wb + vp * ub) / xpb
    exp_4 = (-kappa **2 * qb + kappa * up*pb - up*vp*rb - vp**2 * qb) / kappa
    return exp_3,exp_4

def generateCoords(fovHor,fovVer,start,end):
    xCoords = np.arange(start,end,0.1)
    yCoordsMax = np.tan(np.radians(fovHor/2)) * end
    zCoordsMax = -np.tan(np.radians(fovVer/2)) * end
    yCoords = np.arange(0,yCoordsMax,0.1)
    zCoords = np.arange(zCoordsMax,0,0.1)
    Coords = [xCoords,yCoords,zCoords]
    allCoords = list(itertools.product(*Coords))
    filteredCoords = []
    for pos in allCoords:
        horCheck = np.degrees(np.abs(np.arctan(pos[1]/pos[0])))
        verCheck = np.degrees(np.abs(np.arctan(pos[2]/pos[0])))
        if(horCheck <= fovHor/2 and verCheck<=fovVer/2):
            filteredCoords.append(pos)

    return filteredCoords
def calculateQ(theta,dt,M_theta1s,Mq,g,X_theta1s):
    theta_1s = (-Mq/M_theta1s)*theta
    q = theta/dt
    u_dot = X_theta1s * theta_1s - g*theta
    u = u_dot*dt
    return q,u


def evaluate1DOF_u(attitude,posCop,k,ub,allCoords):
    
    allUpDot_t =[]
    allVpDot_t =[]
    allUpDot_r =[]
    allVpDot_r =[]
    for pos in allCoords:
        posNew = position_viewing(np.radians(attitude[0]),np.radians(attitude[1]),np.radians(attitude[2]),pos,posCop)
        u_p = up(k,posNew[1],posNew[0])
        v_p = vp(k,posNew[1],posNew[0])
        u_p_dot_t,u_p_dot_r = up_dot_comp(k,ub,0,posNew[0],0,0,0,u_p,v_p)
        v_p_dot_t,v_p_dot_r = vp_dot_comp(k,ub,0,v_p,posNew[0],0,0,0,u_p)
        allUpDot_t.append(u_p_dot_t)
        allVpDot_t.append(v_p_dot_t)
        allUpDot_r.append(u_p_dot_r)
        allVpDot_r.append(v_p_dot_r)
    return np.array(allUpDot_t),np.array(allUpDot_r),np.array(allVpDot_t),np.array(allVpDot_r)

def evaluate1DOF_u_k(FoV,screen_size,attitude,posCop,ub,allCoords):
    k = kappa(screen_size,FoV)
    allUpDot_t =[]
    allVpDot_t =[]
    allUpDot_r =[]
    allVpDot_r =[]
    for pos in allCoords:
        posNew = position_viewing(np.radians(attitude[0]),np.radians(attitude[1]),np.radians(attitude[2]),pos,posCop)
        u_p = up(k,posNew[1],posNew[0])
        v_p = vp(k,posNew[1],posNew[0])
        u_p_dot_t,u_p_dot_r = up_dot_comp(k,ub,0,posNew[0],0,0,0,u_p,v_p)
        v_p_dot_t,v_p_dot_r = vp_dot_comp(k,ub,0,v_p,posNew[0],0,0,0,u_p)
        allUpDot_t.append(u_p_dot_t)
        allVpDot_t.append(v_p_dot_t)
        allUpDot_r.append(u_p_dot_r)
        allVpDot_r.append(v_p_dot_r)
    return np.array(allUpDot_t),np.array(allUpDot_r),np.array(allVpDot_t),np.array(allVpDot_r)

def evaluate1DOF_theta(attitude,posCop,k,allCoords):
    allUpDot_t =[]
    allVpDot_t =[]
    allUpDot_r =[]
    allVpDot_r =[]
    q,u = calculateQ(np.radians(attitude[0]),1,26.4,-1.8955,9.81,-9.280)
    for pos in allCoords:
        posNew = position_viewing(np.radians(attitude[0]),np.radians(attitude[1]),np.radians(attitude[2]),pos,posCop)
        u_p = up(k,posNew[1],posNew[0])
        v_p = vp(k,posNew[1],posNew[0])
        u_p_dot_t,u_p_dot_r = up_dot_comp(k,0,0,posNew[0],0,q,0,u_p,v_p)
        v_p_dot_t,v_p_dot_r = vp_dot_comp(k,0,0,v_p,posNew[0],0,q,0,u_p)
        allUpDot_t.append(u_p_dot_t)
        allVpDot_t.append(v_p_dot_t)
        allUpDot_r.append(u_p_dot_r)
        allVpDot_r.append(v_p_dot_r)
    return np.array(allUpDot_t),np.array(allUpDot_r),np.array(allVpDot_t),np.array(allVpDot_r),q
def evaluate1DOF_theta_k(FoV,screen_size,attitude,posCop,allCoords):
    k = kappa(screen_size,FoV)
    allUpDot_t =[]
    allVpDot_t =[]
    allUpDot_r =[]
    allVpDot_r =[]
    q,u = calculateQ(np.radians(attitude[0]),1,26.4,-1.8955,9.81,-9.280)
    for pos in allCoords:
        posNew = position_viewing(np.radians(attitude[0]),np.radians(attitude[1]),np.radians(attitude[2]),pos,posCop)
        u_p = up(k,posNew[1],posNew[0])
        v_p = vp(k,posNew[1],posNew[0])
        u_p_dot_t,u_p_dot_r = up_dot_comp(k,0,0,posNew[0],0,q,0,u_p,v_p)
        v_p_dot_t,v_p_dot_r = vp_dot_comp(k,0,0,v_p,posNew[0],0,q,0,u_p)
        allUpDot_t.append(u_p_dot_t)
        allVpDot_t.append(v_p_dot_t)
        allUpDot_r.append(u_p_dot_r)
        allVpDot_r.append(v_p_dot_r)
    return np.array(allUpDot_t),np.array(allUpDot_r),np.array(allVpDot_t),np.array(allVpDot_r),q

def optimise(upDot_t,upDot_r,vpDot_t,vpDot_r,velocity,q,positions):
    if(velocity ==0):
        velocity =1
    modulus_t = np.sqrt(upDot_t**2 + vpDot_t**2)
    ratio_t = np.abs(modulus_t) / velocity
    print(ratio_t)
    if(q !=0):
        qz = [q*z for x,y,z in positions]
        qx = [-q*x for x,y,z in positions]
        q_total = np.sqrt(np.array(qz) **2 + np.array(qx)**2)
        modulus_r = np.sqrt(upDot_r**2 + vpDot_r**2)
        ratio_r = np.abs(modulus_r) / q_total
    else:
        modulus_r = np.zeros_like(modulus_t)
        ratio_r=np.zeros_like(ratio_t)
    print(ratio_r)
    ratio = ratio_t+ratio_r
    maxVal = max(ratio)
    maxIndex = list(ratio).index(maxVal)
    bestPos = positions[maxIndex]
    maxVal_t = ratio_t[maxIndex]
    maxVal_r = ratio_r[maxIndex]
    return bestPos, maxVal,maxVal_t,maxVal_r
  


def evaluate2DOF(attitude,posCop,k,ub,qb,positions):
    allUpDot_t =[]
    allVpDot_t =[]
    allUpDot_r =[]
    allVpDot_r =[] 
    for pos in positions:
        posNew = position_viewing(np.radians(attitude[0]),np.radians(attitude[1]),np.radians(attitude[2]),pos,posCop)
        u_p = up(k,posNew[1],posNew[0])
        v_p = vp(k,posNew[1],posNew[0])
        u_p_dot_t,u_p_dot_r = up_dot_comp(k,ub,0,posNew[0],0,qb,0,u_p,v_p)
        v_p_dot_t,v_p_dot_r = vp_dot_comp(k,ub,0,v_p,posNew[0],0,qb,0,u_p)
        allUpDot_t.append(u_p_dot_t)
        allVpDot_t.append(v_p_dot_t)
        allUpDot_r.append(u_p_dot_r)
        allVpDot_r.append(v_p_dot_r)
    return np.array(allUpDot_t),np.array(allUpDot_r),np.array(allVpDot_t),np.array(allVpDot_r)

def evaluate2DOF_k(screen_size,FoV,attitude,posCop,ub,qb,positions):
    k = kappa(screen_size,FoV)
    allUpDot_t =[]
    allVpDot_t =[]
    allUpDot_r =[]
    allVpDot_r =[] 
    for pos in positions:
        posNew = position_viewing(np.radians(attitude[0]),np.radians(attitude[1]),np.radians(attitude[2]),pos,posCop)
        u_p = up(k,posNew[1],posNew[0])
        v_p = vp(k,posNew[1],posNew[0])
        u_p_dot_t,u_p_dot_r = up_dot_comp(k,ub,0,posNew[0],0,qb,0,u_p,v_p)
        v_p_dot_t,v_p_dot_r = vp_dot_comp(k,ub,0,v_p,posNew[0],0,qb,0,u_p)
        allUpDot_t.append(u_p_dot_t)
        allVpDot_t.append(v_p_dot_t)
        allUpDot_r.append(u_p_dot_r)
        allVpDot_r.append(v_p_dot_r)
    return np.array(allUpDot_t),np.array(allUpDot_r),np.array(allVpDot_t),np.array(allVpDot_r)





    


#%%
resolution = [7680,2160]
fov_limits = np.radians([140,102]) # [deg] https://risa2000.github.io/hmdgdb/
PPI = 400 # https://community.openmr.com/t/sde-and-ppi-pimax-vs-valve-index-vs-others/20032
screen_size = np.array([resolution[0]/PPI, resolution[1]/PPI]) * 0.0254 # [m]
screen_size_limit=screen_size
k = [screen_size[0] / (2* np.tan(fov_limits[0]/2)) , screen_size[1] / (2* np.tan(fov_limits[1]/2))]
max_limits = [k[0] * np.tan(fov_limits[0]/2), k[1] * np.tan(fov_limits[1]/2)]

# %%
fovRange = np.arange(20,160,20)
allPositions=[]
for angle in fovRange:
    allCoords = generateCoords(angle,np.degrees(fov_limits[1]),1,10)
    allPositions.append(allCoords)
#%%
uRange = np.arange(1,7,1)
posCop = [0,0,0]
attitude1DOF_u=[0,0,0]

maxValsAll=[]
bestHorAll=[]
bestVerAll=[]




#%%
# Case 1: only u, k constant
C1 = pd.DataFrame({})
C1.insert(0,"FoV",np.array(fovRange))
col=1

for ub in uRange:
    bestFOVHor1DOF_u = []
    bestFOVVer1DOF_u =[]
    maxVals1DOF_u = []
    maxVals1DOF_ut = []
    maxVals1DOF_ur = []
    bestPos_u =[]

    for positions in allPositions:
        # 1DOF
        upDot_t,upDot_r,vpDot_t,vpDot_r= evaluate1DOF_u(attitude1DOF_u,posCop,k[0],ub,positions)
        bestPos,maxVal,maxVal_t,maxVal_r = optimise(upDot_t,upDot_r,vpDot_t,vpDot_r,ub,0,positions)
        print(bestPos)
        fovHor = 2 * np.arctan(bestPos[1]/bestPos[0]) * (180/np.pi)
        fovVer = 2 * np.arctan(bestPos[2]/bestPos[0]) * (180/np.pi)
        print(fovVer,fovHor)
        bestFOVHor1DOF_u.append(fovHor)
        bestFOVVer1DOF_u.append(-fovVer)
        maxVals1DOF_u.append(maxVal)
        maxVals1DOF_ut.append(maxVal_t)
        maxVals1DOF_ur.append(maxVal_r)
        bestPos_u.append(str(bestPos))
    maxValsAll.append(maxVals1DOF_u)
    bestHorAll.append(bestFOVHor1DOF_u)
    bestVerAll.append(bestFOVVer1DOF_u)
    C1.insert(col,"Horizontal optimal"+str(ub),np.array(bestFOVHor1DOF_u))
    col+=1
    C1.insert(col,"Vertical optimal"+str(ub),np.array(bestFOVVer1DOF_u))
    col+=1
    C1.insert(col, "maxVal"+str(ub), np.array(maxVals1DOF_u))
    col+=1
C1.to_csv("Data/SensCase_1.csv")
#%%
thetaRange= np.arange(-6,0,1)
attitude1DOF_theta=[-5,0,0]
#%%
# Case 2: only theta, k is constant 
C2=pd.DataFrame({})
C2.insert(0,"FoV",np.array(fovRange))
col=1
for theta in thetaRange:
    attitude1DOF_theta=[theta,0,0]
    bestFOVHor1DOF_theta = []
    bestFOVVer1DOF_theta =[]
    maxVals1DOF_theta = []
    maxVals1DOF_thetat = []
    maxVals1DOF_thetar = []
    bestPos_theta=[]
    for positions in allPositions:
        upDot_t,upDot_r,vpDot_t,vpDot_r,q_done = evaluate1DOF_theta(attitude1DOF_theta,posCop,k[0],positions)
        bestPos,maxVal,maxVal_t,maxVal_r = optimise(upDot_t,upDot_r,vpDot_t,vpDot_r,0,q_done,positions)
        print(bestPos)
        fovHor = 2 * np.arctan(bestPos[1]/bestPos[0]) * (180/np.pi)
        fovVer = 2 * np.arctan(bestPos[2]/bestPos[0]) * (180/np.pi)
        print(fovVer,fovHor)
        bestFOVHor1DOF_theta.append(fovHor)
        bestFOVVer1DOF_theta.append(-fovVer)
        maxVals1DOF_theta.append(maxVal)
        maxVals1DOF_thetat.append(maxVal_t)
        maxVals1DOF_thetar.append(maxVal_r)
        bestPos_theta.append(str(bestPos))    

    C2.insert(col,"Horizontal optimal"+str(theta),np.array(bestFOVHor1DOF_theta))
    col+=1
    C2.insert(col,"Vertical optimal"+str(theta),np.array(bestFOVVer1DOF_theta))
    col+=1
    C2.insert(col, "maxVal"+str(theta), np.array(maxVals1DOF_theta))
    col+=1
C2.to_csv("Data/SensCase_2.csv")
#%%   
# Case 3: 2 DOF, k constant
C3 = pd.DataFrame({})
C3.insert(0,"FoV",np.array(fovRange))
col=1
for u in uRange:
    for theta in thetaRange:
        attitude2DOF=[theta,0,0]
        bestFOVHor2DOF = []
        bestFOVVer2DOF =[]
        bestPos_2DOF =[]
        maxVals2DOF = [] 
        maxVals2DOF_t = [] 
        maxVals2DOF_r = [] 
        for positions in allPositions:
            #2DOF
            q,ub = calculateQ(np.radians(attitude2DOF[0]),1,26.4,-1.8955,9.81,-9.280)
            print(q,u)
            upDot_t,upDot_r,vpDot_t,vpDot_r= evaluate2DOF(attitude2DOF,posCop,k[0],u,q,positions)
            bestPos2DOF,maxVal2DOF,maxVal_t,maxVal_r = optimise(upDot_t,upDot_r,vpDot_t,vpDot_r,u,q,positions)
            print(bestPos2DOF)
            fovHor = 2 * np.arctan(bestPos2DOF[1]/bestPos2DOF[0]) * (180/np.pi)
            fovVer = 2 * np.arctan(bestPos2DOF[2]/bestPos2DOF[0]) * (180/np.pi)
            print(fovVer,fovHor)
            bestFOVHor2DOF.append(fovHor)
            bestFOVVer2DOF.append(-fovVer)
            maxVals2DOF.append(maxVal2DOF)
            maxVals2DOF_t.append(maxVal_t)
            maxVals2DOF_r.append(maxVal_r)
            bestPos_2DOF.append(str(bestPos2DOF))            

        C3.insert(col,"Horizontal optimal" +str(u) +"_" +str(theta),np.array(bestFOVHor2DOF))
        col+=1
        C3.insert(col,"Vertical optimal"+str(u) +"_" +str(theta),np.array(bestFOVVer2DOF))
        col+=1
        C3.insert(col, "maxVal"+str(u) +"_" +str(theta), np.array(maxVals2DOF))
        col+=1
C3.to_csv("Data/SensCase_3.csv")
    


#%%
# Case 4: u and k changes
C4=pd.DataFrame({})
C4.insert(0,"FoV",fovRange)
col =1
for ub in uRange:
    bestFOVHor1DOF_uk = []
    bestFOVVer1DOF_uk =[]
    maxVals1DOF_uk = []
    maxVals1DOF_ukt = []
    maxVals1DOF_ukr = []
    bestPos_uk =[]
    for i in range(0,len(fovRange)):
        angle = fovRange[i]
        positions = allPositions[i]
        upDot_t,upDot_r,vpDot_t,vpDot_r= evaluate1DOF_u_k(angle,screen_size[0],attitude1DOF_u,posCop,ub,positions)
        bestPos,maxVal,maxVal_t,maxVal_r = optimise(upDot_t,upDot_r,vpDot_t,vpDot_r,ub,0,positions)
        print(bestPos)
        fovHor = 2 * np.arctan(bestPos[1]/bestPos[0]) * (180/np.pi)
        fovVer = 2 * np.arctan(bestPos[2]/bestPos[0]) * (180/np.pi)
        print(fovVer,fovHor)
        bestFOVHor1DOF_uk.append(fovHor)
        bestFOVVer1DOF_uk.append(-fovVer)
        maxVals1DOF_uk.append(maxVal)
        maxVals1DOF_ukt.append(maxVal_t)
        maxVals1DOF_ukr.append(maxVal_r)
        bestPos_uk.append(str(bestPos))

    C4.insert(col,"Horizontal optimal"+str(ub),bestFOVHor1DOF_uk)
    col+=1
    C4.insert(col,"Vertical optimal"+str(ub),bestFOVVer1DOF_uk)
    col+=1
    C4.insert(col, "maxVal"+str(ub), maxVals1DOF_uk)
    col+=1
C4.to_csv("Data/SensCase_4.csv")

#%%
# Case 5: q and k changes
C5 = pd.DataFrame({})
C5.insert(0,"FoV",fovRange)
col =1
for theta in thetaRange:
    attitude1DOF_theta=[theta,0,0]
    bestFOVHor1DOF_thetak = []
    bestFOVVer1DOF_thetak =[]
    maxVals1DOF_thetak = []
    maxVals1DOF_thetakt = []
    maxVals1DOF_thetakr = []
    bestPos_thetak =[]
    for i in range(0,len(fovRange)):
        angle = fovRange[i]
        positions = allPositions[i]
        upDot_t,upDot_r,vpDot_t,vpDot_r,q_done = evaluate1DOF_theta_k(angle,screen_size[0],attitude1DOF_theta,posCop,positions)
        bestPos,maxVal,maxVal_t,maxVal_r = optimise(upDot_t,upDot_r,vpDot_t,vpDot_r,0,q_done,positions)
        print(bestPos)
        fovHor = 2 * np.arctan(bestPos[1]/bestPos[0]) * (180/np.pi)
        fovVer = 2 * np.arctan(bestPos[2]/bestPos[0]) * (180/np.pi)
        print(fovVer,fovHor)
        bestFOVHor1DOF_thetak.append(fovHor)
        bestFOVVer1DOF_thetak.append(-fovVer)
        maxVals1DOF_thetak.append(maxVal) 
        maxVals1DOF_thetakt.append(maxVal_t)
        maxVals1DOF_thetakr.append(maxVal_r)
        bestPos_thetak.append(str(bestPos))     

    C5.insert(col,"Horizontal optimal"+str(theta),bestFOVHor1DOF_thetak)
    col+=1
    C5.insert(col,"Vertical optimal"+str(theta),bestFOVVer1DOF_thetak)
    col+=1
    C5.insert(col, "maxVal"+str(theta), maxVals1DOF_thetak)
    col+=1

C5.to_csv("Data/SensCase_5.csv") 
#%%
# Case 6: 2 dof and k changes
C6=pd.DataFrame({})
C6.insert(0,"FoV",fovRange)
col=1
for u in uRange:
    for theta in thetaRange:
        attitude2DOF=[theta,0,0]
        bestFOVHor2DOF_k = []
        bestFOVVer2DOF_k =[]
        maxVals2DOF_k = [] 
        maxVals2DOF_kt = []
        maxVals2DOF_kr = []
        bestPos2DOF_k =[]
        for i in range(0,len(fovRange)):
            angle = fovRange[i]
            positions = allPositions[i]
            q,ub = calculateQ(np.radians(attitude2DOF[0]),1,26.4,-1.8955,9.81,-9.280)
            print(q,u)
            upDot_t,upDot_r,vpDot_t,vpDot_r=evaluate2DOF_k(screen_size[0],angle,attitude2DOF,posCop,u,q,positions)
            bestPos2DOF,maxVal2DOF,maxVal_t,maxVal_r = optimise(upDot_t,upDot_r,vpDot_t,vpDot_r,u,q,positions)
            print(bestPos2DOF)
            fovHor = 2 * np.arctan(bestPos2DOF[1]/bestPos2DOF[0]) * (180/np.pi)
            fovVer = 2 * np.arctan(bestPos2DOF[2]/bestPos2DOF[0]) * (180/np.pi)
            print(fovVer,fovHor)
            bestFOVHor2DOF_k.append(fovHor)
            bestFOVVer2DOF_k.append(-fovVer)
            maxVals2DOF_k.append(maxVal2DOF)
            maxVals2DOF_kt.append(maxVal_t)
            maxVals2DOF_kr.append(maxVal_r)
            bestPos2DOF_k.append(str(bestPos2DOF))    

        C6.insert(col,"Horizontal optimal"+str(u) +"_" +str(theta),bestFOVHor2DOF_k)
        col+=1
        C6.insert(col,"Vertical optimal"+str(u) +"_" +str(theta),bestFOVVer2DOF_k)
        col+=1
        C6.insert(col, "maxVal"+str(u) +"_" +str(theta), maxVals2DOF_k)
        col+=1

C6.to_csv("Data/SensCase_6.csv")
# %%
