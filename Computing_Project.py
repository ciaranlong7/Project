import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import time

#Defining Variables (Must reference these values in my report)
G = 6.6743e-11
M_sun = 1.989e30
S_B_constant = 5.670374419e-8
c = 299792458
h = 6.62607015e-34
k = 1.380649e-23
def R_g(M):
    return (G*(M*M_sun))/(c**2)

def T(r, M, M_dot):
    return ((G*(M*M_sun)*M_dot)/(8*np.pi*((r*R_g(M))**3)*S_B_constant))**(1/4)

#Taking viscous forces into account
#Used to throw up warning - RuntimeWarning: invalid value encountered in double_scalars
#Reason for warning - When I define Rs, the first R value is slightly smaller than the original r_in value
#This leads to r_in/R > 1 and hence fn is negative and (fn)**(1/4) gives an 'invalid value'
def T_visc(r, r_in, M, M_dot):
    return (((3*G*(M*M_sun)*M_dot)/(8*np.pi*((r**3)*((R_g(M))**3))*S_B_constant))*(1-((r_in/(r))**(1/2))))**(1/4)

def T_vs_R(r_in, r_out, M, M_dot, bins):
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    new_Rs = []
    Ts = []
    Ts_visc = []
    for i in range(len(Rs)-1):
        midpoint_r = (Rs[i+1] + Rs[i])/2
        new_Rs.append(midpoint_r)
        Ts.append(T(midpoint_r, M, M_dot)) #divide by 1e6 to scale if desired
        Ts_visc.append(T_visc(midpoint_r, r_in, M, M_dot))
    return Ts, Ts_visc

#Defining luminosity per unit frequency per unit area
#Throwing up warning - RuntimeWarning: overflow encountered in exp
#It's ok to ignore this warning (still performs calculation, just takes time to do).
#See https://www.statology.org/runtimewarning-overflow-encountered-in-exp
def F_v(T, v):
    return ((2*np.pi*h*(v**3)/(c**2))/(np.exp((h*v)/(k*T))-1))

#Plot f_v as a fn of T for multiple different vs.
def plot_f_v(r_in, r_out, M, M_dot, bins):
    #The following two lists must be the same length. If they aren't tweak first for loop below
    small_vs = [1e14, 1e15, 1e16]
    large_vs = [1e17, 1e18, 1e19]

    Ts, Ts_visc = T_vs_R(r_in, r_out, M, M_dot, bins)
        
    new_Ts = np.linspace(min(Ts_visc), max(Ts_visc), bins)
    
    #List of lists with the inner lists being the f_v values across different T values
    smallf_vs = []
    largef_vs = []
    
    for i in range(len(small_vs)):
        small_list = []
        large_list = []
        for T in new_Ts:
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
        ax1.plot(new_Ts, smallf_vs[i], label = f"Frequency = {small_vs[i]:.1e}")
        ax2.plot(new_Ts, largef_vs[i], label = f"Frequency = {large_vs[i]:.1e}")

    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.set_xlabel('T / K')
    ax1.set_ylabel('$log_{10}$(F_$\\nu$ / W $Hz^{-1}$ $m^{-1}$)')
    ax2.set_ylabel('$log_{10}$(F_$\\nu$ / W $Hz^{-1}$ $m^{-1}$)')
    ax1.set_title('F_$\\nu$ against T at different frequency values')
    ax1.legend(loc = 'best')
    ax2.legend(loc = 'best')
    return plt.show()

#Now need to calculate integral. To do this, I must first calculate the integrand of eq 8.3
#I can then evaluate the integral using the trapezium rule.
def integrand(r, r_in, v, M, M_dot):
    t = T_visc(r, r_in, M, M_dot) #Change to T(R) to ignore viscous forces
    f_v = F_v(t, v)
    return f_v*4*np.pi*r*((R_g(M))**2)

#Now evaluate integral with trapezium rule:
def L_v(r_in, r_out, v, M, M_dot, bins):
    
    my_integrands = []
    new_Rs = []
    
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    
    for i in range(len(Rs)-1):
        # Integrand calculated at the midpoint of each bin because at r_in, T_visc=0 giving a div0 error
        midpoint_r = (Rs[i+1] + Rs[i])/2
        new_Rs.append(midpoint_r)
        my_integrand = integrand(midpoint_r, r_in, v, M, M_dot)
        my_integrands.append(my_integrand)
        
    #total sum of all trapeziums
    l_v = trapezoid(my_integrands, x=new_Rs)
        
    return l_v, my_integrands

def spectrum(r_in, r_out, v_start, v_fin, M, M_dot, bins): #Note, same bins for logspacing v and L_v as easier. May not want this though.
    log_vs = []
    spectrum = []
    log_vl_v = []
    
    vs = np.logspace(np.log10(v_start), np.log10(v_fin), bins)
    
    for v in vs:
        log_vs.append(np.log10(v))
        l_v, my_integrands = L_v(r_in, r_out, v, M, M_dot, bins) # R_g units
        spectrum.append(l_v)
        log_vl_v.append(np.log10(v*l_v))
    
    return spectrum, vs, log_vs, log_vl_v

#spectrum is luminosity spectrum (luminosity at all frequencies in 10^14 - 10^19 range)

def plot_spectrum():
    spec, vs, log_vs, log_vl_v = spectrum(6, 10**5, 1e14, 1e19, 10, 10**15, 1000)
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
def T_vs_T_visc(r_in, r_out, M, M_dot, bins):

    Ts, Ts_visc = T_vs_R(r_in, r_out, M, M_dot, bins)
    
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
    print(f"{max(Ts_visc):.2e} K")
    
    return plt.show()

#Now convergence testing
def convergence_check(r_in, r_out, v_start, v_fin, M, M_dot, steps):
    start = time.time()
    vs = np.logspace(np.log10(v_start), np.log10(v_fin), steps)
    
    bins_test = [500, 1000, 2000, 5000, 20000]

    ref_spectrum, vs, log_vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, bins)
    ref_tot = trapezoid(spec, x=vs)

    # #Regular y axis
    # log_bins = []
    # total_ratios = []
    # all_spectrums = np.array(ref_spectrum)
    # for bins in bins_test:
    #     my_list = []
    #     log_bins.append(np.log10(bins))
        # spec, vs, log_vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, bins)
    #     for v in vs:
    #         l_v, my_integrands = L_v(r_in, r_out, v, M, M_dot, bins)
    #         my_list.append(l_v)
    #     tot = trapezoid(my_list, x=vs)
    #     total_ratios.append(tot/ref_tot)
    #     all_spectrums = np.vstack((all_spectrums, my_list))

    # normalised_spectrums = all_spectrums/all_spectrums[0, :]
    # counter = -1
    # for row in normalised_spectrums:
    #     if counter == -1:
    #         plt.plot(log_vs, row, label = "Reference Spectrum - 10000 bins")
    #     else:
    #         plt.plot(log_vs, row, label = f"{bins_test[counter]} bins")
    #     counter += 1
    
    # plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    # plt.ylabel('$\\frac{L_{v}}{L_{v}(Ref)}$')
    # plt.title('Convergence testing for L_v')
    # plt.legend(loc = 'best')
    
    # # Delta L y axis
    # log_bins = []
    # total_ratios = []
    # norm_delta_Ls = np.array([0]*len(ref_spectrum))
    # for bins in bins_test:
    #     norm_delta_L = []
    #     my_list = []
    #     log_bins.append(np.log10(bins))
    #     for i in range(len(vs)):
    #         l_v, my_integrands = L_v(r_in, r_out, vs[i], M, M_dot, bins)
    #         my_list.append(l_v)
    #         norm_delta_L.append((l_v-ref_spectrum[i])/ref_spectrum[i])
    #     tot = trapezoid(my_list, x=vs)
    #     total_ratios.append(tot/ref_tot)
    #     norm_delta_Ls  = np.vstack((norm_delta_Ls, norm_delta_L))
        
    # plt.figure()
    # counter = -1
    # for row in norm_delta_Ls:
    #     if counter == -1:
    #         plt.plot(log_vs, row, label = "Reference Spectrum - 10000 bins")
    #     else:
    #         plt.plot(log_vs, row, label = f"{bins_test[counter]} bins")
    #     counter += 1
    
    # plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    # plt.ylabel('$\\frac{\u0394L_{v}}{L_{v}(Ref)}$')
    # plt.title('Different y-axis scale - Convergence testing for L_v')
    # plt.legend(loc = 'best')

    # plt.figure()
    # plt.plot(log_bins, total_ratios)
    # plt.plot(log_bins, [1,1,1,1,1]) #This represents the reference spectrum of 10000 bins
    # plt.ylim(0.99994, 1.00001)
    # plt.xlabel('$log_{10}$(No of bins)')
    # plt.ylabel('$\\frac{Total L}{Total L(Ref)}$')
    # plt.title('Convergence testing for Total L')

    # spectrum_fixed_v = []
    # for bins in bins_test:
    #     l_v, my_integrands = L_v(r_in, r_out, 1e17, M, M_dot, bins) #1e17 is ref v
    #     spectrum_fixed_v.append(l_v)

    # fixed_vs = [1e14, 1e15, 1e16, 1e17, 1e18, 1e19]

    # spectrum_fixed_v = np.array(spectrum_fixed_v)
    # for v in fixed_vs:
    #     fixed_v_list = []
    #     for bins in bins_test:
    #         l_v, my_integrands = L_v(r_in, r_out, v, M, M_dot, bins)
    #         fixed_v_list.append(l_v)
    #     spectrum_fixed_v = np.vstack((spectrum_fixed_v, fixed_v_list))

    # normalised_spectrum_fixed_v = spectrum_fixed_v/spectrum_fixed_v[0, :]

    # plt.figure()
    # counter = -1
    # for row in normalised_spectrum_fixed_v:
    #     if counter == -1:
    #         plt.plot(log_bins, row, label = "Reference Spectrum - 1e17 Hz")
    #     else:
    #         plt.plot(log_bins, row, label = f"{fixed_vs[counter]:e} Hz")
    #     counter += 1

    # plt.xlabel('$log_{10}$(No of bins)')
    # plt.ylabel('$\\frac{L_{v}}{L_{v}(Ref)}$')
    # plt.title('Convergence testing for L_v (varying v)')
    # plt.legend(loc = 'best')

    end = time.time()
    print(end - start)

    return plt.show()

#What have I done wrong with the L_v/L_v(Ref) plot?

#r_ins is a list of varying r_in values
def spectrum_vary_rin(r_ins, r_out, v_start, v_fin, M, M_dot, bins):
    for r_in in r_ins:
        spec, vs, log_vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, bins) #r_out constant. Usually at 10^5 R_g
        tot = trapezoid(spec, x=vs)
        plt.plot(log_vs, spec, label = f"$r_{{in}}$ = {r_in}$R_{{g}}$, $L_{{Tot}}$ = {tot:.1e}W")
    
    plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    plt.ylabel('$L_{v}$ / W')
    plt.title('Spectrum across $10^{14}$ - $10^{19}$ Hz with varying r_in')
    plt.legend(loc = 'best')
    return plt.show()

def T_visc_vary_rin(r_ins, r_out, M, M_dot, bins):
    for r_in in r_ins:
        Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
        new_Rs = []
        Ts_visc = []
        for i in range(len(Rs)-1):
            midpoint_r = (Rs[i+1] + Rs[i])/2
            new_Rs.append(midpoint_r)
            Ts_visc.append(T_visc(midpoint_r, r_in, M, M_dot)/1e6)
        plt.plot(np.log10(new_Rs), Ts_visc, label = f"$r_{{in}}$ = {r_in}$R_{{g}}$, {round(max(Ts_visc), 1)}$*10^6$K")
    
    plt.xlabel('$log_{10}$($\\frac{R}{R_{g}}$)')
    plt.ylabel('T(R) / $10^6$K')
    plt.title('Viscous forces temp as a function of $log_{10}$(R) with varying r_in')
    plt.legend(loc = 'best')
    
    return plt.show()

#Ms is a list of varying M values
def spectrum_vary_M(r_in, r_out, v_start, v_fin, Ms, M_dot, bins):
    for M in Ms:
        spec, vs, log_vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, bins)
        tot = trapezoid(spec, x=vs)
        plt.plot(log_vs, spec, label = f"M = {M}$M_{{sun}}$, $L_{{Tot}}$ = {tot:.1e}W")

    plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    plt.ylabel('$L_{v}$ / W')
    plt.title('Spectrum across $10^{14}$ - $10^{19}$ Hz with varying M')
    plt.legend(loc = 'best')
    return plt.show()

def T_visc_vary_M(r_in, r_out, Ms, M_dot, bins):
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    for M in Ms:
        new_Rs = []
        Ts_visc = []
        for i in range(len(Rs)-1):
            midpoint_r = (Rs[i+1] + Rs[i])/2
            new_Rs.append(midpoint_r)
            Ts_visc.append(T_visc(midpoint_r, r_in, M, M_dot)/1e6)
        plt.plot(np.log10(new_Rs), Ts_visc, label = f"M = {M}$M_{{sun}}$, $T_{{Max}}$ = {round(max(Ts_visc), 1)}$*10^6$K")
    
    plt.xlabel('$log_{10}$($\\frac{R}{R_{g}}$)')
    plt.ylabel('T(R) / $10^6$K')
    plt.title('Viscous forces temp as a function of $log_{10}$(R) with varying M')
    plt.legend(loc = 'best')
    
    return plt.show()

def spectrum_vary_Mdot(r_in, r_out, v_start, v_fin, M, M_dots, bins):
    for M_dot in M_dots:
        spec, vs, log_vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, bins)
        tot = trapezoid(spec, x=vs)
        plt.plot(log_vs, spec, label = f"$M_{{dot}}$ = {M_dot} Kg$s^{{-1}}$, $L_{{Tot}}$ = {tot:.1e}W")

    plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    plt.ylabel('$L_{v}$ / W')
    plt.title('Spectrum across $10^{14}$ - $10^{19}$ Hz with varying M')
    plt.legend(loc = 'best')
    return plt.show()

def T_visc_vary_Mdot(r_in, r_out, M, M_dots, bins):
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    for M_dot in M_dots:
        new_Rs = []
        Ts_visc = []
        for i in range(len(Rs)-1):
            midpoint_r = (Rs[i+1] + Rs[i])/2
            new_Rs.append(midpoint_r)
            Ts_visc.append(T_visc(midpoint_r, r_in, M, M_dot)/1e6)
        plt.plot(np.log10(new_Rs), Ts_visc, label = f"$M_{{dot}}$ = {M_dot} Kg$s^{{-1}}$, $T_{{Max}}$ = {round(max(Ts_visc), 1)}$*10^6$K")
    
    plt.xlabel('$log_{10}$($\\frac{R}{R_{g}}$)')
    plt.ylabel('T(R) / $10^6$K')
    plt.title('Viscous forces temp as a function of $log_{10}$(R) with varying M')
    plt.legend(loc = 'best')
    
    return plt.show()

r_ins = [1.23, 6, 100, 1000]
Ms = [0.1, 1, 10, 100]
M_dots = [10**14, 10**15, 10**16]
r_in = 6
r_out = 10**5
v_start = 1e14
v_fin = 1e19
M = 10
M_dot = 10**15
bins = 1000
spectrum_vary_Mdot(r_in, r_out, v_start, v_fin, M, M_dots, bins)