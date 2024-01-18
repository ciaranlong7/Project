import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

#Defining Variables (Milestone Problem)
G = 6.6743e-11
M_sun = 1.989e30
M = 10*M_sun
M_dot = 10**15
S_B_constant = 5.670374419e-8
c = 299792458
R_g = (G*M)/(c**2)
h = 6.62607015e-34
k = 1.380649e-23

def T(r):
    return ((G*M*M_dot)/(8*np.pi*((r*R_g)**3)*S_B_constant))**(1/4)

#Taking viscous forces into account
#Used to throw up warning - RuntimeWarning: invalid value encountered in double_scalars
#Reason for warning - When I define Rs, the first R value is slightly smaller than the original r_in value
#This leads to r_in/R > 1 and hence fn is negative and (fn)**(1/4) gives an 'invalid value'
def T_visc(r, r_in):
    return (((3*G*M*M_dot)/(8*np.pi*((r**3)*(R_g**3))*S_B_constant))*(1-((r_in/(r))**(1/2))))**(1/4)

#Defining luminosity per unit frequency per unit area
#Throwing up warning - RuntimeWarning: overflow encountered in exp
#It's ok to ignore this warning (still performs calculation, just takes time to do).
#See https://www.statology.org/runtimewarning-overflow-encountered-in-exp
def F_v(T,v):
    return ((2*np.pi*h*(v**3)/(c**2))/(np.exp((h*v)/(k*T))-1))

#Now need to calculate integral. To do this, I must first calculate the integrand of eq 8.3
#I can then evaluate the integral using the trapezium rule.
def integrand(r, r_in, v):
    t = T_visc(r, r_in) #Change to T(R) to ignore viscous forces
    f_v = F_v(t, v)
    return f_v*4*np.pi*r*(R_g**2)

#Now evaluate integral with trapezium rule:
def L_v(r_in, r_out, v, bins):
    
    my_integrands = []
    new_Rs = []
    
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    
    for i in range(len(Rs)-1):
        # Integrand calculated at the midpoint of each bin because at r_in, T_visc=0 giving a div0 error
        midpoint_r = (Rs[i+1] + Rs[i])/2
        new_Rs.append(midpoint_r)
        my_integrand = integrand(midpoint_r, r_in, v)
        my_integrands.append(my_integrand)
        
    #total sum of all trapeziums
    l_v = trapezoid(my_integrands, x=new_Rs)
        
    return l_v, my_integrands

def spectrum(r_in, r_out):
    log_vs = []
    spectrum = []
    log_vl_v = []
    v_start = 1e14
    v_fin = 1e19
    bins = 1000
    
    vs = np.logspace(np.log10(v_start), np.log10(v_fin), bins)
    
    for v in vs:
        log_vs.append(np.log10(v))
        l_v, my_integrands = L_v(r_in, r_out, v, 1000) # R_g units
        spectrum.append(l_v)
        log_vl_v.append(np.log10(v*l_v))
    
    return spectrum, vs, log_vs, log_vl_v

#spectrum is luminosity spectrum (luminosity at all frequencies in 10^14 - 10^19 range)

def plot_spectrum():
    spec, vs, log_vs, log_vl_v = spectrum(6, 10**5)
    plt.plot(log_vs, log_vl_v)
    # plt.tick_params(axis='both', color = 'white')
    plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    plt.ylabel('$log_{10}$($\\nu$*$L_{v}$ / HzW)')
    # plt.xticks(color = 'white', fontsize = 12)
    # plt.yticks(color = 'white', fontsize = 12)
    plt.title('Spectrum across $10^{14}$ - $10^{19}$ Hz frequency range')

    #total luminosity from the system
    tot = trapezoid(spec, x=vs)
    print("Total luminosity from the system:")
    print(tot)
    
    return plt.show()

#Plotting the difference between T and T_visc
def T_vs_T_visc():
    bins = 1000
    r_in = 6
    r_out = 10**5
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    new_Rs = []
    Ts = []
    Ts_visc = []
    for i in range(len(Rs)-1):
        midpoint_r = (Rs[i+1] + Rs[i])/2
        new_Rs.append(midpoint_r)
        Ts.append(T(midpoint_r)) #divide by 1e6 to scale if desired
        Ts_visc.append(T_visc(midpoint_r, 6))
    
    plt.plot(np.log10(new_Rs), Ts, label = 'No viscous forces considered')
    plt.plot(np.log10(new_Rs), Ts_visc, label = 'Viscous forces considered')
    # plt.tick_params(axis='both', color = 'white')
    plt.xlabel('$log_{10}$($\\frac{R}{R_{g}}$)')
    plt.ylabel('T(R) / $10^6$K')
    # plt.xticks(color = 'white', fontsize = 12)
    # plt.yticks(color = 'white', fontsize = 12)
    plt.title('Temperature as a function of $log_{10}$(Radius)')
    plt.legend(loc = 'upper right')
    
    print("Max temperature with viscous forces considered:")
    print(f"{max(Ts_visc):.6e} K")
    
    return plt.show()

#Plot f_v as a fn of T for multiple different vs.
def plot_f_v():
    #The following two lists must be the same length. If they aren't tweak first for loop below
    small_vs = [1e14, 1e15, 1e16]
    large_vs = [1e17, 1e18, 1e19]
    Ts = np.linspace(min(Ts_visc), max(Ts_visc), 1000)
    
    #List of lists with the inner lists being the f_v values across different T values
    smallf_vs = []
    largef_vs = []
    
    for i in range(len(small_vs)):
        small_list = []
        large_list = []
        for T in Ts:
            smallf_v = F_v(T, small_vs[i])
            largef_v = F_v(T, large_vs[i])
            #Get the following warning - RuntimeWarning: divide by zero encountered in log10
            #Caused by taking log10 of a very small f_v value. Computer treats this as taking log10(0)
            small_list.append(np.log10(smallf_v))
            large_list.append(np.log10(largef_v))
        smallf_vs.append(small_list)
        largef_vs.append(large_list)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)    
    for i in range(len(smallf_vs)):
        ax1.plot(Ts, smallf_vs[i], label = f"Frequency = {small_vs[i]:.1e}")
        ax2.plot(Ts, largef_vs[i], label = f"Frequency = {large_vs[i]:.1e}")

    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.set_xlabel('T / K')
    ax1.set_ylabel('$log_{10}$(F_$\\nu$ / W $Hz^{-1}$ $m^{-1}$)')
    ax2.set_ylabel('$log_{10}$(F_$\\nu$ / W $Hz^{-1}$ $m^{-1}$)')
    ax1.set_title('F_$\\nu$ against T at different frequency values')
    ax1.legend(loc = 'best')
    ax2.legend(loc = 'best')
    return plt.show()

#Now convergence testing
def convergence_check():
    v_start = 1e14
    v_fin = 1e19
    steps = 1000
    vs = np.logspace(np.log10(v_start), np.log10(v_fin), steps)
    fixed_vs = [1e14, 1e15, 1e16, 1e17, 1e18, 1e19]
    
    bins_test = [500, 1000, 2000, 5000, 20000]
    
    #Array containing all spectrums, with each spectrum being different due to a different number of bins
    all_spectrums = []
    total_ratios = []
    
    #Reference L_v
    for v in vs:
        l_v, my_integrands = L_v(r_in, r_out, v, 10000) #10000 is ref bins
        all_spectrums.append(l_v)

    #Total L from reference spectrum
    ref_tot = trapezoid(all_spectrums, x=vs)
    all_spectrums = np.array(all_spectrums)
    
    #Vary bins to check for convergence
    log_bins = []
    for bins in bins_test:
        my_list = []
        log_vs = []
        log_bins.append(np.log10(bins))
        for v in vs:
            log_vs.append(np.log10(v))
            l_v, my_integrands = L_v(r_in, r_out, v, bins)
            my_list.append(l_v)
        #Total luminosity from spectrum/Total luminosity (ref specctrum). (my_list is the spectrum here.)
        tot = trapezoid(my_list, x=vs)
        total_ratio = tot/ref_tot
        total_ratios.append(total_ratio)
        all_spectrums = np.vstack((all_spectrums, my_list))

    # Filling an array delta_L which is (L_v-L_v(ref))/L_v(ref)
    delta_L = np.empty((1,len(all_spectrums[0, :])))
    x = 0
    for row in all_spectrums:
        extra_list = []
        for i in range(len(row)):
            extra_list.append(row[i]-all_spectrums[0, i])
        if x == 0:
            delta_L = np.array(extra_list)
            x += 1
        else:
            delta_L = np.vstack((delta_L, extra_list))
        
    #Normalising with respect to reference value
    normalised_spectrums = all_spectrums/all_spectrums[0, :]
    normalised_delta_L = delta_L/all_spectrums[0, :]
    
    counter = -1
    for row in normalised_spectrums:
        if counter == -1:
            plt.plot(log_vs, row, label = "Reference Spectrum - 10000 bins")
        else:
            plt.plot(log_vs, row, label = f"{bins_test[counter]} bins")
        counter += 1
    
    plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    plt.ylabel('$\\frac{L_{v}}{L_{v}(Ref)}$')
    plt.title('Convergence testing for L_v')
    plt.legend(loc = 'best')

    #Creating a new figure to display convergence test for total L
    plt.figure()
    plt.plot(log_bins, total_ratios)
    plt.plot(log_bins, [1,1,1,1,1]) #This represents the reference spectrum of 10000 bins
    plt.xlabel('$log_{10}$(No of bins)')
    plt.ylabel('$\\frac{Total L}{Total L(Ref)}$')
    plt.title('Convergence testing for Total L')

    spectrum_fixed_v = []
    for bins in bins_test:
        l_v, my_integrands = L_v(r_in, r_out, 1e17, bins) #1e17 is ref v
        spectrum_fixed_v.append(l_v)
    
    spectrum_fixed_v = np.array(spectrum_fixed_v)
    for v in fixed_vs:
        fixed_v_list = []
        for bins in bins_test:
            l_v, my_integrands = L_v(r_in, r_out, v, bins)
            fixed_v_list.append(l_v)
        spectrum_fixed_v = np.vstack((spectrum_fixed_v, fixed_v_list))

    normalised_spectrum_fixed_v = spectrum_fixed_v/spectrum_fixed_v[0, :]

    plt.figure()
    counter_fixed = -1
    for row in normalised_spectrum_fixed_v:
        if counter_fixed == -1:
            plt.plot(log_bins, row, label = "Reference Spectrum - 1e17 Hz")
        else:
            plt.plot(log_bins, row, label = f"{fixed_vs[counter_fixed]} Hz")
        counter_fixed += 1

    plt.xlabel('$log_{10}$(No of bins)')
    plt.ylabel('$\\frac{L_{v}}{L_{v}(Ref)}$')
    plt.title('More convergence testing for L_v')
    plt.legend(loc = 'best')

    plt.figure()
    z = -1
    for row in normalised_delta_L:
        if z == -1:
            plt.plot(log_vs, row, label = "Reference Spectrum - 10000 bins")
        else:
            plt.plot(log_vs, row, label = f"{bins_test[z]} bins")
        z += 1
    
    plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    plt.ylabel('$\\frac{\u0394L_{v}}{L_{v}(Ref)}$')
    plt.title('Different y-axis to display convergence testing for L_v')
    plt.legend(loc = 'best')
    
    return plt.show()

#What have I done wrong with the L_v/L_v(Ref) plot?

#r_ins is a list of varying r_in values
def spectrum_vary_rin(r_ins):
    for r_in in r_ins:
        spec, vs, log_vs, log_vl_v = spectrum(r_in, 10**5) #r_out constant at 10^5 R_g
        tot = trapezoid(spec, x=vs)
        plt.plot(log_vs, spec, label = f"$r_{{in}}$ = {r_in}$R_{{g}}$, Total L = {tot:.1e}")
    
    plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    plt.ylabel('$log_{10}$($\\nu$*$L_{v}$ / HzW)')
    plt.title('Spectrum across $10^{14}$ - $10^{19}$ Hz frequency range with varying r_in')
    plt.legend(loc = 'best')
    return plt.show()

r_ins = [1.23, 100, 1000, 10000]