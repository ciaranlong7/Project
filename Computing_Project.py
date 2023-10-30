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

#Creating list of R values between 6R_g and 10**5R_g
r_in = 6*R_g
r_out = 10**5*R_g

def T(R):
    return ((G*M*M_dot)/(8*np.pi*(R**3)*S_B_constant))**(1/4)

#Defining luminosity per unit frequency per unit area
def F_v(T,v): 
    return ((2*np.pi*h*(v**3)/(c**2))/(np.exp((h*v)/(k*T))-1))

#Now need to calculate integral. To do this, I must first calculate the integrand of eq 8.3
#I can then evaluate the integral using the trapezium rule.

def integrand(R, v):
    t = T(R)
    f = F_v(t,v)
    return f*4*np.pi*R

#Now evaluate integral with trapezium rule (check implementation of trapezium rule):
def L_v(r_in, r_out, v):
    bins = 1000
    span = r_out - r_in
    increment = span/bins
    #sum of all trapeziums
    total = 0
    Rs = []
    Ls = []
    #k = (np.log10(r_out) - np.log10(r_in))/(bins)
    #k is for log R
    
    for i in range(bins):
        #r = r_in*(10**(k*(i+1)))
        r = r_in + i*increment
        Rs.append(r)
        L_v = integrand(r, v)
        Ls.append(L_v)
        
    for i in range(len(Rs)-1):
        delta_R = (Rs[i+1]-Rs[i])
        rectangle = delta_R*Ls[i]
        triangle = delta_R*(Ls[i+1]-Ls[i])/2
        trapezium = rectangle + triangle
        total += trapezium
        
    return total, Rs, Ls

#Temporarily let v = 1e16
v = 1e16

total, Rs, Ls = L_v(r_in, r_out, v)

plt.plot(Rs, Ls)
plt.xlabel('Radius')
plt.ylabel('Luminosity')
plt.title('L(R) Graph')
plt.show()

#Next to do - plot L against log(R) instead of just R