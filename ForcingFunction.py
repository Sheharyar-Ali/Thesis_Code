import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
random.seed(1)


k =[-2,4,-2,1,2,1,1,1]
omega = [0.7,1.5,1,4,2.5,1.8,2.3,4.6]
time = np.arange(0,50,0.1)
ff = np.zeros_like(time)
for  i in range(0,len(k)):
    ff += k[i] * np.sin(omega[i] * time)
ff_dot = np.zeros_like(ff)
for i in range(1,len(ff)):
    ff_dot[i] = (ff[i] - ff[i-1])/(time[i] - time[i-1])

ff_ranges = []
