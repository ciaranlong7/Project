import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

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

#Now evaluate integral with trapezium rule:
def L_v(r_in, r_out, v):
    bins = 1000
    span = r_out - r_in
    increment = span/bins
    
    ys = []
    
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    
    for r in Rs:
        y = integrand(r, v)
        ys.append(y)
        
    #total sum of all trapeziums
    l_v = trapezoid(ys, x=Rs, dx=increment)
        
    return l_v, Rs, ys

def spectrum(r_in, r_out):
    log_vs = []
    spectrum = []
    log_vl_v = []
    v_start = 10e14
    v_fin = 10e19
    bins = 1000
    span = v_fin - v_start
    increment = span/bins
    
    vs = np.logspace(np.log10(v_start), np.log10(v_fin), bins)
    
    for v in vs:
        log_vs.append(np.log(v))
        l_v, Rs, ys = L_v(r_in, r_out, v)
        spectrum.append(l_v)
        log_vl_v.append(np.log10(v*l_v))
    
    return spectrum, vs, log_vs,log_vl_v

spectrum, vs, log_vs, log_vl_v = spectrum(r_in, r_out)

tot = trapezoid(spectrum, x=vs, dx=increment)

plt.plot(log_vl_v, log_vs)
plt.xlabel('log(v / Hz)')
plt.ylabel('log(v*L_v / HzW)')
plt.title('log(freq*luminosity)/ log(frequency) Graph (all base 10)')

#Can also plot the non log graph