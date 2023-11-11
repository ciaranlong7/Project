import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

#Defining Variables (Milestone Problem)
G = 6.6743e-11
M_sun = 1.989e30
M = 10*M_sun
M_dot = 10e15
S_B_constant = 5.670374419e-8
c = 299792458
R_g = (G*M)/(c**2)
h = 6.62607015e-34
k = 1.380649e-23

#Creating list of R values between 6R_g and 10**5R_g
r_in = 6*R_g
r_out = (10**5)*R_g

def T(R):
    return ((G*M*M_dot)/(8*np.pi*(R**3)*S_B_constant))**(1/4)

#Taking viscous forces into account
#Used to throw up warning - RuntimeWarning: invalid value encountered in double_scalars
#Reason for warning - When I define Rs, the first R value is slightly smaller than the original r_in value
#This leads to r_in/R > 1 and hence fn is negative and (fn)**(1/4) gives an 'invalid value'
def T_visc(R):
    return (((3*G*M*M_dot)/(8*np.pi*(R**3)*S_B_constant))*(1-((r_in/R)**(1/2))))**(1/4)

#Defining luminosity per unit frequency per unit area
#Throwing up warning - RuntimeWarning: overflow encountered in exp
#It's ok to ignore this warning (still performs calculation, jut takes time to do).
#See https://www.statology.org/runtimewarning-overflow-encountered-in-exp
def F_v(T,v):
    return ((2*np.pi*h*(v**3)/(c**2))/(np.exp((h*v)/(k*T))-1))

#Now need to calculate integral. To do this, I must first calculate the integrand of eq 8.3
#I can then evaluate the integral using the trapezium rule.
def integrand(R, v):
    t = T_visc(R) #Change to T(R) to ignore viscous forces
    f = F_v(t,v)
    return f*4*np.pi*R

#Now evaluate integral with trapezium rule:
def L_v(r_in, r_out, v):
    bins = 1000
    span = r_out - r_in
    increment = span/bins
    
    ys = []
    new_Rs = []
    
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    
    for i in range(len(Rs)-1):
        # Temperature calculated at the midpoint of each bin because at r_in T_visc=0 giving a div0 error
        midpoint_r = (Rs[i+1] + Rs[i])/2
        new_Rs.append(midpoint_r)
        y = integrand(midpoint_r, v)
        ys.append(y)
        
    #total sum of all trapeziums
    l_v = trapezoid(ys, x=new_Rs, dx=increment)
        
    return l_v, Rs, ys

def spectrum(r_in, r_out):
    log_vs = []
    spectrum = []
    log_vl_v = []
    v_start = 1e14
    v_fin = 1e19
    bins = 1000
    span = v_fin - v_start
    increment = span/bins
    
    vs = np.logspace(np.log10(v_start), np.log10(v_fin), bins)
    
    for v in vs:
        log_vs.append(np.log10(v))
        l_v, Rs, ys = L_v(r_in, r_out, v)
        spectrum.append(l_v)
        log_vl_v.append(np.log10(v*l_v))
    
    return spectrum, vs, log_vs, log_vl_v

#spectrum is luminosity spectrum (luminosity at all frequencies in 10^14 - 10^19 range)
spectrum, vs, log_vs, log_vl_v = spectrum(r_in, r_out)

#total luminosity from the system
#Should be roughly 7x10^30. I get about 7x10^31.
v_start = 10e14
v_fin = 10e19
bins = 1000
span = v_fin - v_start
increment = span/bins
tot = trapezoid(spectrum, x=vs, dx=increment)
print("Total luminosity from the system:")
print(tot)

plt.plot(log_vs, log_vl_v)
plt.xlabel('log10(v / Hz)')
plt.ylabel('log10(v*L_v / HzW)')
plt.title('log10(v*L_v) against log10(v)')
plt.show()

plt.plot(vs, spectrum)
plt.xlabel('v / Hz')
plt.ylabel('L_v / W')
plt.title('Spectrum across defined frequency range')
plt.show()

bins_ = 1000
Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins_)
new_Rs = []
Ts = []
Ts_visc = []
for i in range(len(Rs)-1):
    midpoint_r = (Rs[i+1] + Rs[i])/2
    new_Rs.append(midpoint_r)
    Ts.append(T(midpoint_r))
    Ts_visc.append(T_visc(midpoint_r))
    
plt.plot(np.log10(new_Rs), Ts, label = 'no viscous forces considered')
plt.plot(np.log10(new_Rs), Ts_visc, label = 'viscous forces considered')
plt.xlabel('log(R / m)')
plt.ylabel('T(R) / K')
plt.title('Temperature as a function of R (log x-axis scale)')
plt.legend(loc = 'upper right')
plt.show()