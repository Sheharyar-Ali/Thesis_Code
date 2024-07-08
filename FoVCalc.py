#%%
import numpy as np
import matplotlib.pyplot as plt
import itertools

def depressionAngle(z,x):
    return np.arctan(z/x)

def depressionRate(z_dot,z,depression,x_dot):
    attitudeChange = (z_dot/z) * np.cos(depression) * np.sin(depression)
    positionChange = -1 * (x_dot/z) * np.sin(depression) * np.sin(depression)
    return attitudeChange, positionChange

def splayAngle(y,z):
    return np.arctan(y/z)

def splayRate(z_dot,z,splay,y_dot):
    altitudeChange = -1 * (z_dot/z) * np.cos(splay) * np.sin(splay)
    positionChange = (y_dot/z) * np.cos(splay) * np.cos(splay)
    return altitudeChange,positionChange

def create_ground_points(x_range, y_range, z_height, num_points):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, z_height)
    points_world = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return points_world


def position_viewing(theta, phi, psi, pos_world, pos_COP):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), np.sin(phi)],
                   [0, -np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, -np.sin(theta)],
                   [0, 1, 0],
                   [np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), np.sin(psi), 0],
                   [-np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    pos_viewing = R @ (np.array(pos_world) - np.array(pos_COP))
    return pos_viewing

def project_to_retina(pos_viewing, focal_length):
    x, y, z = pos_viewing
    if x != 0:
        u = focal_length * (y / x)
        v = -focal_length * (z / x)
    else:
        u, v = float('inf'), float('inf')  # Avoid division by zero
    return np.array([u, v])


def optic_flow(points_world, observer_position, observer_orientation, V, focal_length, dt):
    theta, phi, psi = observer_orientation
    optic_flows = []

    for pos_world in points_world:
        # Transform world coordinates to observer coordinates
        pos_obs = position_viewing(theta, phi, psi, pos_world, observer_position)
        
        # Project current position to retina
        current_proj = project_to_retina(pos_obs, focal_length)
        
        # Calculate the new observer position after moving forward
        new_observer_position = observer_position + np.array([V * dt, 0, 0])
        
        # Transform world coordinates to the new observer coordinates
        new_pos_obs = position_viewing(theta, phi, psi, pos_world, new_observer_position)
        
        # Project new position to retina
        new_proj = project_to_retina(new_pos_obs, focal_length)
        
        # Calculate optic flow as the change in retina coordinates
        flow = new_proj - current_proj
        optic_flows.append((current_proj, flow))
    
    return optic_flows
def optic_flow_pitching(points_world, observer_position, omega_pitch, focal_length, dt):
    optic_flows = []

    for pos_world in points_world:
        # Assume no translation, phi and psi are 0
        pos_obs = position_viewing(omega_pitch * dt, 0, 0, pos_world, observer_position)
        
        # Project current position to retina
        current_proj = project_to_retina(pos_obs, focal_length)
        
        # Calculate the new observer position after time dt
        new_pos_obs = position_viewing(omega_pitch * dt * 2, 0, 0, pos_world, observer_position)
        
        # Project new position to retina
        new_proj = project_to_retina(new_pos_obs, focal_length)
        
        # Calculate optic flow as the change in retina coordinates
        flow = new_proj - current_proj
        optic_flows.append((current_proj, flow))
    
    return optic_flows
def optic_flow_combined(points_world, observer_position, omega_pitch, V, focal_length, dt):
    optic_flows = []

    for pos_world in points_world:
        # Transform world coordinates to observer coordinates with pitching and translation
        pos_obs = position_viewing(omega_pitch * dt, 0, 0, pos_world, observer_position)
        pos_obs += np.array([V * dt, 0, 0])  # Translation along x-axis
        
        # Project current position to retina
        current_proj = project_to_retina(pos_obs, focal_length)
        
        # Calculate the new observer position after time dt
        new_pos_obs = position_viewing(omega_pitch * dt * 2, 0, 0, pos_world, observer_position)
        new_pos_obs += np.array([V * dt * 2, 0, 0])  # Translation along x-axis
        
        # Project new position to retina
        new_proj = project_to_retina(new_pos_obs, focal_length)
        
        # Calculate optic flow as the change in retina coordinates
        flow = new_proj - current_proj
        optic_flows.append((current_proj, flow))
    
    return optic_flows


def generateCoords(fovHor,fovVer,start,end):
    xCoords = np.arange(start,end,0.1)
    yCoordsMax = np.tan(np.radians(fovHor/2)) * end
    yCoords = np.arange(0,yCoordsMax,0.1)
    zCoords = np.array([5])
    Coords = [xCoords,yCoords,zCoords]
    allCoords = list(itertools.product(*Coords))
    
    filteredCoords = []
    for pos in allCoords:
        horCheck = np.degrees(np.abs(np.arctan(pos[1]/pos[0])))
        verCheck = np.degrees(np.abs(np.arctan(pos[2]/pos[0])))
        if(horCheck <= fovHor/2 and verCheck<=fovVer/2):
            filteredCoords.append(pos)
    return filteredCoords


# %%
fovRange = np.arange(20,160,20)
fovVer = 100
allCoords =[]
for angle in fovRange:
    newCoords = generateCoords(fovHor=angle,fovVer=fovVer,start=1,end=10)
    allCoords.append(newCoords)
print(allCoords)
# %%
# Case 1: Only translation
x_range = (0, 10)  # X range on the ground
y_range = (0, 27)  # Y range on the ground
z_height = -5  # Z height for ground plane
num_points = 20  # Number of points in each dimension
points_world = create_ground_points(x_range, y_range, z_height, num_points)
u = 2
dt = 0.1
focalLength = 0.017
obsPos = [0,0,0]
obsOrient=[0,0,0]
flows = optic_flow(points_world, obsPos, obsOrient, u, focalLength, dt)
plt.figure(figsize=(10, 10))
for current_proj, flow in flows:
    if np.isfinite(current_proj).all() and np.isfinite(flow).all():
        plt.arrow(current_proj[0], current_proj[1], flow[0], flow[1], head_width=0.01, head_length=0.01, color='blue')

plt.xlabel('Retina X Coordinate (m)')
plt.ylabel('Retina Y Coordinate (m)')
plt.title('Optic Flow Pattern on the Retina for Forward Movement')
plt.grid()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()
#%%
# Case 2 only pitching
omegaPitch = 0.1
# Calculate optic flow for pitching motion
flows_pitching = optic_flow_pitching(points_world, obsPos, omegaPitch, focalLength, dt)

# Plot optic flow on the retina
plt.figure(figsize=(10, 10))
for current_proj, flow in flows_pitching:
    if np.isfinite(current_proj).all() and np.isfinite(flow).all():
        plt.arrow(current_proj[0], current_proj[1], flow[0], flow[1], head_width=0.02, head_length=0.02, color='blue')

plt.xlabel('Retina X Coordinate (m)')
plt.ylabel('Retina Y Coordinate (m)')
plt.title('Optic Flow Pattern on the Retina for Pitching Motion')
plt.grid()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()
#%%
# case 3
# Calculate optic flow for combined pitching and translation motion
flows_combined = optic_flow_combined(points_world, obsPos, omegaPitch, u, focalLength, dt)

# Plot optic flow on the retina
plt.figure(figsize=(10, 10))
for current_proj, flow in flows_combined:
    if np.isfinite(current_proj).all() and np.isfinite(flow).all():
        plt.arrow(current_proj[0], current_proj[1], flow[0], flow[1], head_width=0.02, head_length=0.03, color='blue')

plt.xlabel('Retina X Coordinate (m)')
plt.ylabel('Retina Y Coordinate (m)')
plt.title('Optic Flow Pattern on the Retina for Combined Pitching and Translation Motion')
plt.grid()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()        
        
