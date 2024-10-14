
# %%
# Inner loop
import control
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import get_window

fig_count = 1
M_theta1s = 26.4
M_q = -1.8954
M_u = 0.05
X_theta1s = -9.280
X_u = -0.02
g = 9.80665
X_q = 0.6674
tf_theta1s_theta = M_theta1s * control.tf([1],[1,0])

# Outer loop
ss_tf_theta1s_theta = M_theta1s/-M_q
num_hc2 = (X_theta1s/ ss_tf_theta1s_theta) - g
tf_theta_u = num_hc2 * control.tf([1],[1,-X_u])
print(tf_theta_u)

#%%
Kp = 0.2
w_n = 9 #[rad/s]
damp_n = 0.1 
tau_e = 0.2
# tau_L = -1/X_u
numerator = [1]
denominator = [1]
num_de,den_de = control.pade(tau_e,3)
delay = control.tf(num_de,den_de)
no_delay = control.tf(Kp * np.array(numerator),denominator)
print(no_delay)
tf_pilot_inner_noNM = delay * no_delay
NM = w_n**2 * control.tf([1],[1,w_n*damp_n,w_n**2])
# tf_pilot_inner = tf_pilot_inner_noNM * NM
tf_pilot_inner = tf_pilot_inner_noNM
inner_ol = control.series(tf_pilot_inner,tf_theta1s_theta)
print(inner_ol)
gm, pm, wcg, wcp = control.margin(inner_ol)

print('Crossover freq = ', wcp, ' rad/sec')
print(gm,pm,wcg,wcp)
w_i_inner_calc = (wcp-4.36)/0.18 
w_i_outer = 1/3 * w_i_inner_calc
print("bandwidths inner: %f outer: %f " %(w_i_inner_calc,w_i_outer))


# %%
# Velocity forcing function
# wc = 5
# wi_inner = (wc-4.76)/0.18
# wi_outer = (1/3) * wi_inner
np.random.seed(34)
T_m = 120
T_lead_in = 30
t_m = np.arange(0,T_m,0.01)
w_m = (2*np.pi) / T_m
f_range = [0.1,20]
A_multiplier = 1
#n_d = [5,11,23,37,51,71,101,137,171,226]
n_d = [3, 7, 11, 17, 23, 37, 51, 73, 103, 139]
w_d = np.array(n_d) * w_m
A_d = np.ones_like(w_d) * A_multiplier

for i in range(0,len(A_d)):
    val = A_d[i]
    w = w_d[i]
    if w> w_i_outer:
        A_d[i] = val/10
phi_d = np.random.uniform(-2 * np.pi, 2 * np.pi, 10)
#phi_d =np.array([1.636,4.016,-1.194,-1.731,4.938,5.442,2.274,2.973,3.429,3.486])
phi_d=[-1.5765753638,
       5.6638430155,
       2.9153218531,
       1.2397790767,
       -4.3225972486,
       -4.3229003508,
       -5.5532851101,
       4.6015051579,
       1.2706487122,
       2.6147171]
f_d = []
f_d_lead_in = []
f=np.zeros_like(t_m)
for i in range (0,len(w_d)): 
    buffer =  A_d[i] * np.sin(w_d[i] * t_m  + phi_d[i])
    f_d.append( buffer)
    f+= f_d[i]
f_d_lead_in = -1 * f[0:3000]
f_d_lead_in = f_d_lead_in[::-1]
t = np.arange(0,T_m+T_lead_in,0.01)
f_full = np.concatenate((f_d_lead_in,f))

plt.figure(fig_count)
fig_count+=1
plt.bar(w_d,A_d,width=0.1,label="original")
plt.xlabel("Frequency [rad/s]")
plt.ylabel("Amplitude [m/s]")
plt.legend()

plt.figure(fig_count)
fig_count+=1
plt.title("Forcing Function with Lead in time")
plt.ylabel("Velocity [m/s]")
plt.xlabel("Time [s]")
plt.plot(t[0:3001],f_full[0:3001],label="lead in")
plt.plot(t[3000:],f_full[3000:],label="ff")
plt.legend()

plt.figure(fig_count)
fig_count+=1
plt.title("Forcing Function")
plt.ylabel("Velocity [m/s]")
plt.xlabel("Time [s]")
plt.plot(t_m,f)
info = pd.DataFrame({})
info.insert(0,"forcing function",f_full)
info.to_csv("Heli_Sim/Assets/Scripts/forcing_func.csv")

#%%
# Theta forcing function
np.random.seed(1)
phi_d = np.random.uniform(-2 * np.pi, 2 * np.pi, 10)
n_theta = [2, 5, 11, 23, 37, 51, 71, 83, 97, 137]
w_theta = np.array(n_theta) * w_m
A_theta_mult = 5
A_theta =np.ones_like(w_theta) * A_theta_mult
for i in range(0,len(A_theta)):
    val = A_theta[i]
    w = w_theta[i]
    if w> w_i_outer:
        A_theta[i] = val/10
f_theta_d = []
f_theta_lead_in = []
f_theta=np.zeros_like(t_m)
for i in range (0,len(w_d)): 
    buffer =  A_theta[i] * np.sin(w_theta[i] * t_m  + phi_d[i])
    f_theta_d.append(buffer)
    f_theta+= f_theta_d[i]
f_theta_lead_in = -1 * f_theta[0:3000]
f_theta_lead_in = f_theta_lead_in[::-1]
f_theta_full = np.concatenate((f_theta_lead_in,f_theta))
plt.figure(fig_count)
fig_count+=1
plt.title("Forcing Function Theta")
plt.ylabel("Theta [deg]")
plt.xlabel("Time [s]")
# plt.plot(t_m,f_theta)
plt.plot(t[0:3001],f_theta_full[0:3001],label="lead in")
plt.plot(t[3000:],f_theta_full[3000:],label="ff")
plt.legend()

info_theta = pd.DataFrame({})
info_theta.insert(0,"forcing function",f_theta_full)
info_theta.to_csv("Heli_Sim/Assets/Scripts/forcing_func_theta.csv")
#%%
# Training functions
n_d_training_1 = [5,11,23,37,51,71,101,137,171,226]
w_d_training_1 = np.array(n_d_training_1) * w_m
A_d_training_1 = np.ones_like(w_d_training_1) * A_multiplier
for i  in range(0,len(A_d_training_1)):
    val = A_d_training_1[i]
    w = w_d_training_1[i]
    if w> w_i_outer:
        A_d_training_1[i] = val/10
f_d_training_1 = []
f_d_lead_in_training_1 = []
f_training_1=np.zeros_like(t_m)
for i in range (0,len(w_d_training_1)): 
    buffer =  A_d_training_1[i] * np.sin(w_d_training_1[i] * t_m  + phi_d[i])
    f_d_training_1.append( buffer)
    f_training_1+= f_d_training_1[i]
f_d_lead_in_training_1 = -1 * f_training_1[0:3000]
f_d_lead_in_training_1 = f_d_lead_in_training_1[::-1]
f_full_training_1 = np.concatenate((f_d_lead_in_training_1,f_training_1))

n_d_training_2 = [3,11,23,37,51,77,101,137,143,147]
w_d_training_2 = np.array(n_d_training_2) * w_m
A_d_training_2 = np.ones_like(w_d_training_2) * A_multiplier
for i  in range(0,len(A_d_training_2)):
    val = A_d_training_2[i]
    w = w_d_training_2[i]
    if w> w_i_outer:
        A_d_training_2[i] = val/10
f_d_training_2 = []
f_d_lead_in_training_2 = []
f_training_2=np.zeros_like(t_m)
for i in range (0,len(w_d_training_2)): 
    buffer =  A_d_training_2[i] * np.sin(w_d_training_2[i] * t_m  + phi_d[i])
    f_d_training_2.append( buffer)
    f_training_2+= f_d_training_2[i]
f_d_lead_in_training_2 = -1 * f_training_2[0:3000]
f_d_lead_in_training_2 = f_d_lead_in_training_2[::-1]
f_full_training_2 = np.concatenate((f_d_lead_in_training_2,f_training_2))

plt.figure(fig_count)
fig_count+=1
plt.title("TrainingFunction")
plt.ylabel("Velocity [m/s]")
plt.xlabel("Time [s]")
plt.plot(t[0:3001],f_full_training_1[0:3001],label="lead in_1")
plt.plot(t[3000:],f_full_training_1[3000:],label="ff_1")
plt.plot(t[0:3001],f_full_training_2[0:3001],label="lead in_2")
plt.plot(t[3000:],f_full_training_2[3000:],label="ff_2")
plt.legend()

training_info = pd.DataFrame({})
training_info.insert(0,"training_1",f_full_training_1)
training_info.insert(0,"training_2",f_full_training_2)
training_info.to_csv("Heli_Sim/Assets/Scripts/training.csv")
# %%
#Check FFT for leakage
n = len(f)
fs = 100  # Sampling frequency is the inverse of the time step (0.01 seconds)
fft_result = np.fft.fft(f)
fft_freqs = np.fft.fftfreq(n, 1/fs)

# Only take the positive frequencies
fft_result = fft_result[:n // 2]
fft_freqs = fft_freqs[:n // 2]

# Calculate the magnitude
fft_magnitude = np.abs(fft_result) / n

# Plot the FFT result
plt.figure(figsize=(12, 6))
plt.plot(fft_freqs, fft_magnitude)
plt.title("FFT of Forcing Function")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

# %%
