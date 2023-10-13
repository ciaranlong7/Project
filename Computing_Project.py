import numpy as np
import matplotlib.pyplot as plt

#Defining Variables (Milestone Problem)

G = 6.6743e-11
M_sun = 2e30
M = 10*M_sun
M_dot = 10e15
S_B_constant = 5.670374419e-8
c = 299792458
R_g = (G*M)/(c**2)
h = 6.62607015e-34
k = 1.380649e-23

#Creating list of R values between 6 and 10**5
start_R = 6
end_R = 10**5
increment_R = (end_R-start_R)/100000

R = [start_R + i*increment_R for i in range(end_R - start_R)]

def T(R):
    return ((G*M*M_dot)/(8*np.pi*(R**3)*S_B_constant))**(1/4)

#Calculating T Values that correspond to R values
T_R = [T(i*R_g) for i in R]

#Line plot of result (impossible to decipher anything)
# plt.plot(R, T_R)
# plt.xlabel('Radius')
# plt.ylabel('T(R)')
# plt.title('T(R) Graph')

log_R = [np.log(i) for i in R]
T_log_R = [T(i) for i in log_R]

#I notice for this graph, you observe more of a curve as the increment between R values is made smaller
# plt.plot(log_R, T_log_R)
# plt.xlabel('log(Radius)')
# plt.ylabel('T(log(R))')
# plt.title('T(log(R)) Graph')

#Defining frequency range (1e14 to 1e19)
start_v = 6
end_v = 10**5
increment_v = (end_v-start_v)/10000

v = [start_v + i*increment_v for i in range(end_v - start_v)]
#Temporarily let v = 1e16
v = 1e16
#Luminosity per unit frequency per unit area; F DOES NOT correspond to frequency
def F(T_R):
    return [((2*np.pi*h*(v**3)/(c**2))/(np.exp((h*v)/(k*TEMP))-1)) for TEMP in T_R]

#To do - supposed to plot F for a range of frequencies, but how can I plot F for a range of frequencies and temperatures.