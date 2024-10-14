# # %%
# plt.close("all")

# def deltaH(delta,D,theta,n):
#     exp_1 = (D * np.tan(theta)) / (D-delta)
#     delta_h = n * (exp_1 - np.tan(theta))
#     return delta_h
# def v(D,n,theta,delta_dot,dt):
#     delta = delta_dot * dt
#     exp_1 = (D-delta) **2
#     v = D * n * np.tan(theta) * delta_dot * 1/exp_1
#     return v
# def requiredD(deltah_req, theta, n, delta):
#     exp_1 = (n * np.tan(theta)) / deltah_req
#     D = delta + delta* exp_1
#     return D
# def requiredRatio(n,theta,deltaH):
#     exp_1 = np.tan(theta) / deltaH
#     ratio = (1/n) * exp_1
#     return ratio
# def requiredD_delta(n,theta,deltaH):
#     return 1+ ((n* np.tan(theta)) / deltaH)


# # res = [7680x2160]

# delta = 0.1
# D = 6
# theta = 30 * (np.pi/180)
# n = 2

# delta_range = np.arange(0,1.5,0.1)
# theta_range = np.arange(0,90,10)
# theta_range = theta_range * (np.pi / 180)
# D_range = np.linspace(n,12,len(delta_range))
# n_range = np.linspace(0,D,len(delta_range))


# delta_h_theta = deltaH(delta=delta, D = D, theta=theta_range,n=n)
# delta_h_delta = deltaH(delta=delta_range, D = D, theta=theta,n=n)
# delta_h_D = deltaH(delta=delta, D = D_range, theta=theta,n=n)
# delta_h_n = deltaH(delta=delta, D = D, theta=theta,n=n_range)




# # %%
# Required_D = requiredD(deltah_req=delta,theta=theta_range,n=n,delta=delta)
# #Required_ratio = requiredRatio(n=n,theta=theta_range,deltaH=delta)
# Required_DDelta = requiredD_delta(n=n,theta=theta_range,deltaH=delta)

# # %%
# plt.figure(1)
# plt.title("deltaH vs theta for delta=0.1, D=6, n=2")
# plt.ylabel("deltaH")
# plt.xlabel("theta")
# plt.plot(theta_range * (180/np.pi),delta_h_theta)
# plt.hlines(delta,min(theta_range * (180/np.pi)),max(theta_range * (180/np.pi)))

# # %%
# plt.figure(2)
# plt.title("deltaH vs delta for theta=30, D=6, n=2")
# plt.ylabel("deltaH")
# plt.xlabel("delta")
# plt.plot(delta_range,delta_h_delta)

# # %%
# plt.figure(3)
# plt.title("deltaH vs D for delta=0.1,theta = 30, n=2")
# plt.ylabel("deltaH")
# plt.xlabel("D")
# plt.plot(D_range,delta_h_D)

# # %%
# plt.figure(4)
# plt.title("deltaH vs n for delta=0.1, D= 6, theta = 30")
# plt.ylabel("deltaH")
# plt.xlabel("n")
# plt.plot(n_range,delta_h_n)

# # %%
# plt.figure(5)
# plt.title("ratio D/delta needed for deltaH = 0.1, n=2")
# plt.ylabel("D/delta needed")
# plt.xlabel("theta")
# plt.plot(theta_range* (180/np.pi),Required_DDelta)

# # %%
# theta_range = np.arange(0,90,10) * (np.pi / 180)
# plt.figure(6)
# plt.title("velocity percieved over time for different viewing angles")
# plt.plot(time, ff,label="ff")
# plt.xlabel("time")
# plt.ylabel("v percieved")

# for angle in theta_range:
#     plt.plot(time,v(D=D,n=n,theta=angle,delta_dot=ff,dt=0.1),label=(angle * (180/np.pi)))
# plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

# # %%
# plt.figure(7)
# plt.title("Error in perceived velocity")
# plt.xlabel("time")
# plt.ylabel("error in perception")

# for angle in theta_range:
#     plt.plot(time, np.abs(ff - v(D=D,n=n,theta=angle,delta_dot=ff,dt=0.1)),label=(angle * (180/np.pi)))
# plt.legend()
# plt.show()




# plt.figure(1)
# plt.xlabel("x")
# plt.ylabel("screen_v/actual_v")
# plt.title("Ratio of screen velocity of an object and actual velocity of an observer as a function of x world")
# dot_matrix=[]
# dot_matrix_alt =[]
# for position in pos_world_x[0]:
#     pos_world = [position,pos_world_x[1],pos_world_x[2]]
#     modulus = full_calc(pos_world=pos_world,pos_COP=pos_COP,theta=theta,phi=phi,psi=psi,screen_size=screen_size[0],screen_size_limit=screen_size_limit,FoV=fov_limits[0],ub=ub,vb=vb,wb=wb,pb=pb,qb=qb,rb=rb)
#     ratio = modulus/ff
#     dot_matrix.append(ratio[2])
#     modulus_alt = full_calc(pos_world=pos_world,pos_COP=pos_COP,theta=theta,phi=phi,psi=psi,screen_size=screen_size[1],screen_size_limit=screen_size_limit,FoV=fov_limits[1],ub=ub,vb=vb,wb=wb,pb=pb,qb=qb,rb=rb)
#     ratio_alt = modulus_alt/ff
#     dot_matrix_alt.append(ratio_alt[2])
# max_index_x = dot_matrix.index(max(dot_matrix))
# optimal_x = pos_world_x[0][max_index_x]
# plt.plot(pos_world_x[0],dot_matrix,label="Horizontal FoV")
# plt.plot(pos_world_x[0],dot_matrix_alt,label="Vertical FoV")
# plt.grid()
# plt.legend()

# # %%
# # Case 2: changing yw
# y_limits = [k[0] * np.tan(fov_limits[0]/2), 20 * np.tan(fov_limits[0]/2)]
# pos_world_y = [optimal_x,np.arange(y_limits[0],y_limits[1],0.1),random_pos[2]]
# dot_matrix=[]
# dot_matrix_alt = []
# plt.figure(2)
# plt.xlabel("y")
# plt.ylabel("screen_v/actual_v")
# plt.title("Ratio of screen velocity of an object and actual velocity of an observer as a function of y world")

# for y in pos_world_y[1]:
#     pos_world = [pos_world_y[0],y,pos_world_y[2]]
#     modulus = full_calc(pos_world=pos_world,pos_COP=pos_COP,theta=theta,phi=phi,psi=psi,screen_size=screen_size[0],screen_size_limit=screen_size_limit,FoV=fov_limits[0],ub=ub,vb=vb,wb=wb,pb=pb,qb=qb,rb=rb)
#     ratio = modulus/ff
#     dot_matrix.append(ratio[2])
#     modulus_alt = full_calc(pos_world=pos_world,pos_COP=pos_COP,theta=theta,phi=phi,psi=psi,screen_size=screen_size[1],screen_size_limit=screen_size_limit,FoV=fov_limits[1],ub=ub,vb=vb,wb=wb,pb=pb,qb=qb,rb=rb)
#     ratio_alt = modulus_alt/ff
#     dot_matrix_alt.append(ratio_alt[2])
# plt.plot(pos_world_y[1],dot_matrix,label="Horizontal FoV")
# plt.plot(pos_world_y[1], dot_matrix_alt,label="Vertical FoV")
# plt.grid()
# plt.legend()



# # %%
# z_limits = [k[1] * np.tan(fov_limits[1]/2), 20 * np.tan(fov_limits[1]/2)]
# pos_world_z = [random_pos[0],random_pos[1],np.arange(-z_limits[1],-z_limits[0],0.1)]
# dot_matrix=[]
# dot_matrix_alt = []
# plt.figure(3)
# plt.xlabel("z")
# plt.ylabel("screen_v/actual_v")
# plt.title("Change in v perceieved as a function of z world")
# for z in pos_world_z[2]:
#     pos_world = [pos_world_z[0],pos_world_z[1],z]
#     modulus = full_calc(pos_world=pos_world,pos_COP=pos_COP,theta=theta,phi=phi,psi=psi,screen_size=screen_size[0],screen_size_limit=screen_size_limit,FoV=fov_limits[0],ub=ub,vb=vb,wb=wb,pb=pb,qb=qb,rb=rb)
#     ratio = modulus/ff
#     dot_matrix.append(ratio[2])
#     modulus_alt = full_calc(pos_world=pos_world,pos_COP=pos_COP,theta=theta,phi=phi,psi=psi,screen_size=screen_size[1],screen_size_limit=screen_size_limit,FoV=fov_limits[1],ub=ub,vb=vb,wb=wb,pb=pb,qb=qb,rb=rb)
#     ratio_alt = modulus_alt/ff
#     dot_matrix_alt.append(ratio_alt[2])    
# plt.plot(pos_world_z[2],dot_matrix,label="Horizontal FoV")
# plt.plot(pos_world_z[2],dot_matrix_alt,label="Vertical FoV")
# plt.grid()

# Sphere plot
# print("Starting sphere stuff now")
# fig, ax = plt.subplots(figsize=(8, 6))

# initial_azimuths = np.zeros(len(points_world))
# initial_elevations = np.zeros(len(points_world))
# initial_delta_azimuths = np.zeros(len(points_world))
# initial_delta_elevations = np.zeros(len(points_world))

# # Initialize a quiver object with dummy data
# quiver = ax.quiver(initial_azimuths, initial_elevations, initial_delta_azimuths, initial_delta_elevations, color='green')

# textTime = ax.text(0.05, 0.85, "", transform=ax.transAxes)
# textU = ax.text(0.05, 0.95, "", transform=ax.transAxes)
# textTheta = ax.text(0.05, 0.90, "", transform=ax.transAxes)
# ax.set_xlabel('Azimuth (°)')
# ax.set_ylabel('Elevation (°)')
# ax.set_xlim(-120, 120)
# ax.set_ylim(100, 150)


# def updateValues(frame,df,ax):
#     azimuths,elevations,\
#     new_azimuths,new_elevations,\
#     oldPositions,newPositions, newPositionsTrans,newPositionsRot = [],[],[],[],[],[],[],[]
#     u = np.array(df['U'].iloc[frame])
#     theta= np.array(df['Theta'].iloc[frame])
#     time = np.array(df['Time'].iloc[frame])
#     dt = df["Time"].iloc[1] - df["Time"].iloc[0]
#     for point in points_world:
#         old,new,oldPos,newPosRot,newPosTrans,newPos = ProjRetina(observerOrigin,point,r,u,dt,theta)
#         azimuths.append(old[0])
#         elevations.append(old[1])
#         new_azimuths.append(new[0])
#         new_elevations.append(new[1])
#         oldPositions.append(oldPos)
#         newPositions.append(newPos)
#         newPositionsTrans.append(newPosTrans)
#         newPositionsRot.append(newPosRot)
        
#     azimuths = np.array(azimuths)
#     new_azimuths = np.array(new_azimuths)
#     elevations = np.array(elevations)
#     new_elevations = np.array(new_elevations)
#     delta_azimuth = new_azimuths-azimuths
#     delta_elevation = new_elevations -elevations   
#     # print(len(azimuths), len(elevations), len(delta_azimuth), len(delta_elevation))
#     # print(azimuths[0], delta_azimuth[0], new_azimuths[0])
#     # print(delta_elevation)
#     quiver.set_offsets(np.column_stack([azimuths, elevations]))
#     #quiver.set_UVC(delta_azimuth, delta_elevation)
#     quiver.set_UVC(new_azimuths, new_elevations)

#     textU.set_text(f'u: {u}')
#     textTheta.set_text(f'theta: {np.degrees(theta)}')
#     textTime.set_text(f'time:{time}')
#     return quiver, textU, textTheta, textTime
# ## Saving sphere plot
# anime = ani.FuncAnimation(fig,functools.partial(updateValues,df=behaviour,ax=ax),frames=len(t),blit=False)
# anime.save('animation_2d.gif', writer='pillow', fps=60) 
# plt.show()
# exit()