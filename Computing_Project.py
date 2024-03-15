import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import time
import random

#Defining Constants
G = 6.6743e-11
M_sun = 1.989e30
S_B_constant = 5.670374419e-8
c = 299792458
h = 6.62607015e-34
k = 1.380649e-23
m_p = 1.67262192e-27
sigma_T = 6.6524587321e-29
def R_s(M):
    return (2*G*(M*M_sun))/(c**2)

def T(r, M, M_dot):
    return ((G*(M*M_sun)*M_dot)/(8*np.pi*((r*R_s(M))**3)*S_B_constant))**(1/4)

def T_visc(r, r_in, M, M_dot):
    return (((3*G*(M*M_sun)*M_dot)/(8*np.pi*(((r*R_s(M))**3))*S_B_constant))*(1-((r_in/(r))**(1/2))))**(1/4)

#GR Temperature:
def z_1(a_star):
    return 1 + ((1-a_star**2)**(1/3))*((1+a_star)**(1/3)+(1-a_star)**(1/3))

def z_2(a_star):
    return (3*(a_star**2)+(z_1(a_star))**2)**(1/2)

def r_ms_prograde(a_star):
    return 3 + z_2(a_star) - ((3-z_1(a_star))*(3+z_1(a_star)+2*z_2(a_star)))**(1/2)

def r_ms_retrograde(a_star):
    return 3 + z_2(a_star) + ((3-z_1(a_star))*(3+z_1(a_star)+2*z_2(a_star)))**(1/2)

def plot_rms():
    retro_a_stars = np.linspace(-0.998, 0, 101)
    pro_a_stars = np.linspace(0.01, 0.998, 100)
    r_ms_values = []
    for a_star in retro_a_stars:
        r_ms_values.append(r_ms_retrograde(a_star))
    for a_star in pro_a_stars:
        r_ms_values.append(r_ms_prograde(a_star))

    comb_a_stars = np.linspace(-0.998, 0.998, 201)

    plt.plot(comb_a_stars, r_ms_values)
    plt.xlabel('$a_{*}$', fontsize = 16)
    plt.ylabel('$r_{in}/R_{s}$', fontsize = 16)
    # plt.title('How $r_{ms}$ varies with spin parameter $a_{*}$')
    return plt.show()
    
def y_ms_prograde(a_star):
    return (r_ms_prograde(a_star))**(1/2)

def y_ms_retrograde(a_star):
    return (r_ms_retrograde(a_star))**(1/2)

def y(r, M):
    return (r)**(1/2)

def y_1(a_star):
    return 2*np.cos((np.arccos(a_star) - np.pi)/3)

def y_2(a_star):
    return 2*np.cos((np.arccos(a_star) + np.pi)/3)

def y_3(a_star):
    return -2*np.cos((np.arccos(a_star))/3)

def A_prograde(a_star, r, M):
    return 1 - (y_ms_prograde(a_star)/y(r, M)) - (3*(a_star)*np.log(((y(r, M))/(y_ms_prograde(a_star)))))/(2*y(r, M))

def A_retrograde(a_star, r, M):
    return 1 - (y_ms_retrograde(a_star)/y(r, M)) - (3*(a_star)*np.log((y(r, M))/(y_ms_retrograde(a_star))))/(2*y(r, M))

def B_prograde(a_star, r, M):
    result = ((3*((y_1(a_star) - a_star)**2))*np.log(((y(r, M) - y_1(a_star))/(y_ms_prograde(a_star) - y_1(a_star)))))/(y(r, M)*y_1(a_star)*(y_1(a_star)-y_2(a_star))*(y_1(a_star)-y_3(a_star))) \
    + ((3*((y_2(a_star) - a_star)**2))*np.log(((y(r, M) - y_2(a_star))/(y_ms_prograde(a_star) - y_2(a_star)))))/(y(r, M)*y_2(a_star)*(y_2(a_star)-y_1(a_star))*(y_2(a_star)-y_3(a_star))) \
    + ((3*((y_3(a_star) - a_star)**2))*np.log(((y(r, M) - y_3(a_star))/(y_ms_prograde(a_star) - y_3(a_star)))))/(y(r, M)*y_3(a_star)*(y_3(a_star)-y_1(a_star))*(y_3(a_star)-y_2(a_star)))
    return result

def B_retrograde(a_star, r, M):
    result = ((3*((y_1(a_star) - a_star)**2))*np.log(((y(r, M) - y_1(a_star))/(y_ms_retrograde(a_star) - y_1(a_star)))))/(y(r, M)*y_1(a_star)*(y_1(a_star)-y_2(a_star))*(y_1(a_star)-y_3(a_star))) \
    + ((3*((y_2(a_star) - a_star)**2))*np.log(((y(r, M) - y_2(a_star))/(y_ms_retrograde(a_star) - y_2(a_star)))))/(y(r, M)*y_2(a_star)*(y_2(a_star)-y_1(a_star))*(y_2(a_star)-y_3(a_star))) \
    + ((3*((y_3(a_star) - a_star)**2))*np.log(((y(r, M) - y_3(a_star))/(y_ms_retrograde(a_star) - y_3(a_star)))))/(y(r, M)*y_3(a_star)*(y_3(a_star)-y_1(a_star))*(y_3(a_star)-y_2(a_star)))
    return result

def C(a_star, r):
    return 1 - (3/r) + ((2*a_star)/(r**(3/2)))

def T_GR_prograde(r, M, M_dot, a_star):
    return (((3*G*(M*M_sun)*M_dot)/(8*np.pi*((r*R_s(M))**3)*S_B_constant))*((A_prograde(a_star, r, M)-B_prograde(a_star, r, M))/C(a_star, r)))**(1/4)

def T_GR_retrograde(r, M, M_dot, a_star):
    return (((3*G*(M*M_sun)*M_dot)/(8*np.pi*((r*R_s(M))**3)*S_B_constant))*((A_retrograde(a_star, r, M)-B_retrograde(a_star, r, M))/C(a_star, r)))**(1/4)

def T_vs_R(r_out, M, M_dot, R_steps, a_star):
    if a_star >= 0:
        r_in = r_ms_prograde(a_star)
    else:
        r_in = r_ms_retrograde(a_star)
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), R_steps)
    new_Rs = []
    Ts = []
    Ts_visc = []
    Ts_GR = []
    for i in range(len(Rs)-1):
        midpoint_r = (Rs[i+1] + Rs[i])/2
        new_Rs.append(midpoint_r)
        Ts.append(T(midpoint_r, M, M_dot)/1e6) #divide by 1e6 to scale if desired
        Ts_visc.append(T_visc(midpoint_r, r_in, M, M_dot)/1e6)
        if a_star >= 0:
            Ts_GR.append(T_GR_prograde(midpoint_r, M, M_dot, a_star)/1e6)
        else:
            Ts_GR.append(T_GR_retrograde(midpoint_r, M, M_dot, a_star)/1e6)

    return Ts, Ts_visc, Ts_GR, new_Rs

def plot_compare_Ts(r_out, M, M_dot, R_steps, a_star):

    Ts, Ts_visc, Ts_GR, new_Rs = T_vs_R(r_out, M, M_dot, R_steps, a_star)
    
    plt.plot(np.log10(new_Rs), Ts, label = 'Newton - Non viscous')
    plt.plot(np.log10(new_Rs), Ts_visc, label = 'Newton - Viscous')
    plt.plot(np.log10(new_Rs), Ts_GR, label = 'General Relativitiy')
    # plt.tick_params(axis='both', color = 'white')
    plt.xlabel('$log_{10}$($\\frac{R}{R_{s}}$)', fontsize = 16)
    plt.ylabel('T(R) / $10^{6}$K', fontsize = 16)
    # plt.xticks(color = 'white', fontsize = 12)
    # plt.yticks(color = 'white', fontsize = 12)
    # plt.title('Temperature as a function of $log_{10}$(Radius)', fontsize = 14)
    plt.legend(loc = 'upper right')
    
    # print("Max temperature with viscous forces considered:")
    # print(f"{max(Ts_visc):.2e} K")
    
    return plt.show()

def plot_Ts_vary_a_star(r_out, M, M_dot, R_steps, a_stars):
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']
    i=0
    for a_star in a_stars:
        #random_color = [random.random() for _ in range(3)]
        random_color = colours[i]
        Ts, Ts_visc, Ts_GR, new_Rs = T_vs_R(r_out, M, M_dot, R_steps, a_star)
        plt.plot(np.log10(new_Rs), Ts_GR, color = random_color, label = f'General Relativity, $a_{{*}}$={a_star}')
        plt.plot(np.log10(new_Rs), Ts_visc, linestyle='--', color = random_color, label = f'Newton - Viscous')
        i+=1

    plt.xlabel('$log_{10}$($\\frac{R}{R_{s}}$)', fontsize = 16)
    plt.ylabel('T(R) / $10^{6}$K', fontsize = 16)
   # plt.title('Temperature as a function of $log_{10}$(Radius)', fontsize = 14)
    plt.legend(loc = 'upper right')

    return plt.show()

#Defining luminosity per unit frequency per unit area
def F_v(T, v):
    return ((2*np.pi*h*(v**3)/(c**2))/(np.exp((h*v)/(k*T))-1))

#Plot f_v as a fn of T for multiple different vs.
def plot_f_v(r_in, r_out, M, M_dot, bins, a_star):
    #The following two lists must be the same length. If they aren't tweak first for loop below
    small_vs = [1e14, 1e15, 1e16]
    large_vs = [1e17, 1e18, 1e19]

    Ts, Ts_visc, Ts_GR, new_Rs = T_vs_R(r_out, M, M_dot, bins, a_star)
        
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

#Now need to calculate integral. To do this, I must first calculate the integrand of eqn 8.3
#I can then evaluate the integral using the trapezium rule.
#Select which temperature model you like below
def integrand(r, r_in, v, M, M_dot, a_star):
    # t = T(r, M, M_dot)
    # t = T_visc(r, r_in, M, M_dot)
    if a_star>=0:
        t = T_GR_prograde(r, M, M_dot, a_star)
    else:
        t = T_GR_retrograde(r, M, M_dot, a_star)
        
    f_v = F_v(t, v)
    return f_v*4*np.pi*r*((R_s(M))**2)

#Now evaluate integral with trapezium rule:
def L_v(r_in, r_out, v, M, M_dot, bins, a_star):
    
    my_integrands = []
    new_Rs = []
    
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    
    for i in range(len(Rs)-1):
        # Integrand calculated at the midpoint of each bin because at r_in, T_visc=0 giving a div0 error
        midpoint_r = (Rs[i+1] + Rs[i])/2
        new_Rs.append(midpoint_r)
        my_integrand = integrand(midpoint_r, r_in, v, M, M_dot, a_star)
        my_integrands.append(my_integrand)
        
    #total sum of all trapeziums
    l_v = trapezoid(my_integrands, x=new_Rs)
        
    return l_v, my_integrands

#spectrum is luminosity spectrum (luminosity at all frequencies in v_start - v_fin range)
#NOTE: a_star is NOT relevant if you are not using the GR temperature model.
#You select which temperature model you want in the integrand function above
#v_bins is the number of bins used to work out L_Tot with the trapezoid method
def spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star):
    spectrum = []
    log_vl_v = []
    
    vs = np.logspace(np.log10(v_start), np.log10(v_fin), v_bins)
    
    for v in vs:
        l_v, my_integrands = L_v(r_in, r_out, v, M, M_dot, bins, a_star) # R_s units
        spectrum.append(l_v)
        log_vl_v.append(np.log10(v*l_v))
    return spectrum, vs, log_vl_v

def plot_spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star):
    spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star)
    log_vs = np.log10(vs)
    plt.plot(log_vs, log_vl_v)
    # plt.tick_params(axis='both', color = 'white')
    plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    plt.ylabel('$log_{10}$($\\nu$*$L_{\\nu}$ / HzW)')
    # plt.xticks(color = 'white', fontsize = 12)
    # plt.yticks(color = 'white', fontsize = 12)
    plt.title('Spectrum across $10^{14}$ - $10^{19}$ Hz frequency range')

    #total luminosity from the system
    tot = trapezoid(spec, x=vs)
    print("Total luminosity from the system:")
    print(tot)
    
    return plt.show()

#Now convergence testing
def convergence_check(r_in, r_out, v_start, v_fin, v_bins, M, M_dot, a_star):
    start = time.time()
    vs = np.logspace(np.log10(v_start), np.log10(v_fin), v_bins)
    log_vs = np.log10(vs)
    
    bins_test = [500, 1000, 2000, 5000, 20000]
    log_bins = np.log10(bins_test)

    ref_spectrum = []
    
    #Reference L_v
    for v in vs:
        l_v, my_integrands = L_v(r_in, r_out, v, M, M_dot, 10000, a_star) #These are reference parameters
        ref_spectrum.append(l_v)

    #Total L from reference spectrum
    ref_tot = trapezoid(ref_spectrum, x=vs)

    # Regular y axis
    total_ratios = []
    all_spectrums = np.array(ref_spectrum)
    for bins in bins_test:
        my_list = []
        for v in vs:
            l_v, my_integrands = L_v(r_in, r_out, v, M, M_dot, bins, a_star)
            my_list.append(l_v)
        tot = trapezoid(my_list, x=vs)
        total_ratios.append(tot/ref_tot)
        all_spectrums = np.vstack((all_spectrums, my_list))

    normalised_spectrums = all_spectrums/all_spectrums[0, :]
    counter = -1
    for row in normalised_spectrums:
        if counter == -1:
            plt.plot(log_vs, row, label = "Reference Spectrum - 10000 bins")
        else:
            plt.plot(log_vs, row, label = f"{bins_test[counter]} bins")
        counter += 1
    
    plt.xlabel('$log_{10}$($\\nu$ / Hz)')
    plt.ylabel('$\\frac{L_{\\nu}}{L_{\\nu}(ref)}$')
    plt.title('Convergence testing for L_v')
    plt.legend(loc = 'best')
    
    # # Delta L y axis
    # total_ratios = []
    # norm_delta_Ls = np.array([0]*len(ref_spectrum))
    # for bins in bins_test:
    #     norm_delta_L = []
    #     my_list = []
    #     for i in range(len(vs)):
    #         l_v, my_integrands = L_v(r_in, r_out, vs[i], M, M_dot, bins, a_star)
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
    # plt.ylabel('$\\frac{\u0394L_{\\nu}}{L_{\\nu}(ref)}$')
    # plt.title('Different y-axis scale - Convergence testing for L_v')
    # plt.legend(loc = 'best')

    # L_Tot/L_Tot(ref) vs log(number of bins)
    plt.figure()
    plt.plot(log_bins, total_ratios)
    plt.plot(log_bins, [1,1,1,1,1]) #This represents the reference spectrum of 10000 bins
    plt.xlabel('$log_{10}$(No of bins)')
    plt.ylabel('$\\frac{L_{Tot}}{L_{Tot}ref)}$')
    plt.title('Convergence testing for Total L')

    end = time.time()
    print(end - start)

    return plt.show()

#Fixed_vs is a list of fixed v values. Eg: fixed_vs = [1e14, 1e15, 1e16, 1e17, 1e18, 1e19]
def convergence_check_fixed_v(r_in, r_out, fixed_vs, M, M_dot, a_star):
    bins_test = [25, 100, 250, 500, 750, 1000]
    log_bins = np.log10(bins_test)
    
    spectrum_fixed_v = []
    for v in fixed_vs:
        my_list = []
        l_v_ref, my_integrands = L_v(r_in, r_out, v, M, M_dot, 5000, a_star) #These are reference parameters
        for bins in bins_test:
            l_v, my_integrands = L_v(r_in, r_out, v, M, M_dot, bins, a_star)
            my_list.append(l_v/l_v_ref)
        spectrum_fixed_v.append(my_list)

    plt.figure()
    plt.plot(log_bins, [1, 1, 1, 1, 1, 1], label = 'Line representing convergence - 1')
    for i, L_list in enumerate(spectrum_fixed_v):
        plt.plot(log_bins, L_list, label = f"{fixed_vs[i]:.1e} Hz")

    plt.xlabel('$log_{10}$(No of bins)')
    plt.ylabel('$\\frac{L_{\\nu}}{L_{\\nu}(ref)}$')
    plt.title('Convergence testing for L_v (varying v)')
    plt.legend(loc = 'best')
    return plt.show()

#r_ins is a list of varying r_in values
def spectrum_vary_rin(r_ins, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star):
    for r_in in r_ins:
        spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star) #r_out constant. Usually at 10^5 R_s
        log_vs = np.log10(vs)
        tot = trapezoid(spec, x=vs)
        plt.plot(log_vs, [x/1e11 for x in spec], label = f"$r_{{in}}$ = {r_in}$R_{{s}}$, $L_{{Tot}}$ = {tot:.1e}W")

    plt.xlabel('$log_{10}$($\\nu$ / Hz)', fontsize = 16)
    plt.ylabel('$L_{\\nu}$ / $10^{11}$W', fontsize = 16)
    # plt.title(f'Luminosity Spectrum - vary $r_{{in}}$')
    plt.legend(loc = 'best')
    return plt.show()

def spectrum_vary_rin_M_dot(r_ins, r_out, v_start, v_fin, M, M_dots, v_bins, bins, a_star):
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']
    #coded to work with 2 M_dots
    j = 0
    for r_in in r_ins:
        for i, M_dot in enumerate(M_dots):
            spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star) #r_out constant. Usually at 10^5 R_s
            log_vs = np.log10(vs)
            tot = trapezoid(spec, x=vs)
            if i == 0:
                plt.plot(log_vs, [x/1e11 for x in spec], color = colours[j], label = f"$r_{{in}}$ = {r_in}$R_{{s}}$, $L_{{Tot}}$ = {tot:.1e}W")
            else:
                plt.plot(log_vs, [x/1e11 for x in spec], color = colours[j], linestyle = '--', label = f"$r_{{in}}$ = {r_in}$R_{{s}}$, $L_{{Tot}}$ = {tot:.1e}W")

        j+=1
    
    plt.xlabel('$log_{10}$($\\nu$ / Hz)', fontsize = 16)
    plt.ylabel('$L_{\\nu}$ / $10^{11}$W', fontsize = 16)
    # plt.title(f'Luminosity Spectrum - vary $r_{{in}}$')
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
        plt.plot(np.log10(new_Rs), Ts_visc, label = f"$r_{{in}}$ = {r_in}$R_{{s}}$, {round(max(Ts_visc), 1)}$*10^6$K")
    
    plt.xlabel('$log_{10}$($\\frac{R}{R_{s}}$)')
    plt.ylabel('T(R) / $10^6$K')
    plt.title('Viscous forces temp as a function of $log_{10}$(R) with varying $R_{in}$')
    plt.legend(loc = 'best')
    
    return plt.show()

def spectrum_vary_Mdot(r_in, r_out, v_start, v_fin, M, M_dots, v_bins, bins, a_star):
    for M_dot in M_dots:
        spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star)
        log_vs = np.log10(vs)
        tot = trapezoid(spec, x=vs)
        plt.plot(log_vs, [x/1e11 for x in spec], label = f"$\dot{{M}}$ = {M_dot:.0e} Kg$s^{{-1}}$, $L_{{Tot}}$ = {tot:.1e}W")

    plt.xlabel('$log_{10}$($\\nu$ / Hz)', fontsize = 16)
    plt.ylabel('$L_{\\nu}$ / $10^{11}$W', fontsize = 16)
    # plt.title(f'Luminosity Spectrum - vary $\dot{{M}}$')
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
        plt.plot(np.log10(new_Rs), Ts_visc, label = f"$\dot{{M}}$ = {M_dot:.0e} Kg$s^{{-1}}$, $T_{{Max}}$ = {round(max(Ts_visc), 1)}$*10^6$K")
    
    plt.xlabel('$log_{10}$($\\frac{R}{R_{s}}$)')
    plt.ylabel('T(R) / $10^6$K')
    plt.title('Viscous forces temp as a function of $log_{10}$(R) with varying $\dot{M}$')
    plt.legend(loc = 'best')
    
    return plt.show()

#Ms is a list of varying M values
#Now use a GR Temp model
def spectrum_vary_M_a_star(r_out, v_start, v_fin, Ms, M_dot, v_bins, bins, a_stars):
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']
    #coded to for plot to 'work' with only 2 a_stars (still will run just won't look as neat)
    for M in Ms:
        for i, a_star in enumerate(a_stars):
            if a_star >= 0:
                r_in = r_ms_prograde(a_star)
            else:
                r_in = r_ms_retrograde(a_star)
            spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star)
            log_vs = np.log10(vs)
            tot = trapezoid(spec, x=vs)
            if i == 0:
                plt.plot(log_vs, [x/1e11 for x in spec], label = f"M = {M}$M_{{\odot}}$, $a_{{*}}$ = {a_star}, $L_{{Tot}}$ = {tot:.1e}W")
            else:
                plt.plot(log_vs, [x/1e11 for x in spec], linestyle = '--', label = f"M = {M}$M_{{\odot}}$, $a_{{*}}$ = {a_star}, $L_{{Tot}}$ = {tot:.1e}W")

    plt.xlabel('$log_{10}$($\\nu$ / Hz)', fontsize = 16)
    plt.ylabel('$L_{\\nu}$ / $10^{11}$W', fontsize = 16)
    # plt.title(f'Luminosity Spectrum with varying M, $r_{{in}}$={r_in}$R_{{s}}$, $\dot{{M}}$={M_dot:.1e}kg$s^{{-1}}$', fontsize = 9)
    plt.legend(loc = 'best')
    return plt.show()
    
#Old code (uses viscous temp model unlike above)
def T_visc_vary_M(r_in, r_out, Ms, M_dot, bins):
    Rs = np.logspace(np.log10(r_in), np.log10(r_out), bins)
    for M in Ms:
        new_Rs = []
        Ts_visc = []
        for i in range(len(Rs)-1):
            midpoint_r = (Rs[i+1] + Rs[i])/2
            new_Rs.append(midpoint_r)
            Ts_visc.append(T_visc(midpoint_r, r_in, M, M_dot)/1e6)
        plt.plot(np.log10(new_Rs), Ts_visc, label = f"M = {M}$M_{{\odot}}$, $T_{{Max}}$ = {round(max(Ts_visc), 1)}$*10^6$K")
    
    plt.xlabel('$log_{10}$($\\frac{R}{R_{s}}$)')
    plt.ylabel('T(R) / $10^6$K')
    plt.title('Viscous forces temp as a function of $log_{10}$(R) with varying M')
    plt.legend(loc = 'best')
    
    return plt.show()

def L_edd(M):
    return (4*np.pi*G*(M*M_sun)*m_p*c)/sigma_T

#For this function to work you must have selected the general relativistic temperature model.
#Select the model in the integrand function.
def eta_M_dot_vs_a_star(r_out, v_start, v_fin, M, M_dot, v_bins, bins):
    start = time.time()
    a_stars = np.linspace(-0.998, 0.998, 8)
    Eddington_L = L_edd(M)
    
    #calculate efficiencies:
    efficiencies = []
    #Also then calculate how M_dot depends on a_star.
    #Not to be confused with the M_dot input to calculate the efficiency
   # M_dots = []
    for i, a_star in enumerate(a_stars):
        if a_star < 0:
            r_in = r_ms_retrograde(a_star)
        else:
            r_in = r_ms_prograde(a_star)
        spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star)
        tot = trapezoid(spec, x=vs)
        efficiencies.append(tot/Eddington_L)
        # new_M_dot = (Eddington*(c**2))/(efficiency)
        # M_dots.append(new_M_dot)
        print(i)

    end = time.time()
    print(end - start)

    return a_stars, efficiencies#, M_dots

#Theoretical etas from fanidakis 2011 paper
def fanidakis_eta(a_star):
    if a_star >= 0:
        r_in = r_ms_prograde(a_star)
    else:
        r_in = r_ms_retrograde(a_star)
    return 1-np.sqrt(1-((2/3)*(1/r_in)))

def fanidakis():
    a_stars = np.linspace(-0.998, 0.998, 101)
    efficiency = []
    for a_star in a_stars:
        efficiency.append(fanidakis_eta(a_star))
    return efficiency

efficiency = fanidakis()
#a_stars, efficiencies = eta_M_dot_vs_a_star(1e6, 1e12, 1e18, 3, 10**13, 1000, 1000)
#Need to uncomment and call above two lines first
def plot_eta_vs_a_star():
    # Calculate the residuals
    residuals = []
    for i in range(len(efficiencies)):
        residuals.append(efficiencies[i]*100 - efficiency[i])
    
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    # Remove vertical space between axes
    fig.subplots_adjust(hspace=0)
    
    # Plot each graph, and manually set the y tick values
    axs[0].plot(a_stars, efficiency, label = 'Theoretical Method')
    axs[0].plot(a_stars, np.array(efficiencies)*100, label = "This Research's Method")
    axs[0].legend(loc='best')
    axs[0].set_ylabel('\u03B7', fontsize = 16)
    axs[0].set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    
    axs[1].scatter(a_stars, residuals)
    axs[1].plot(a_stars, [0]*len(a_stars), linestyle = '--', color = 'black')
    axs[1].set_xlabel('$a_*$', fontsize = 16)
    axs[1].set_ylabel('$\u03B7_{RM} - \u03B7_{TM}$', fontsize = 16)
    axs[1].set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
    axs[1].set_ylim(-0.16, 0.16)
    
    return plt.show()

#Function built for GR temp for Eddington model since varying a_*. Hence no r_in argument.
def spectrum_edd_vary_a_star(r_out, v_start, v_fin, M, v_bins, bins, test_a_stars):
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']
    Eddington = L_edd(M)
    totals = []
    specs = []
    for i, a_star in enumerate(test_a_stars):
        if a_star >= 0:
            r_in = r_ms_prograde(a_star)
        else:
            r_in = r_ms_retrograde(a_star)
        eta_interpolated = np.interp(a_star, a_stars, efficiency)
        M_dot = Eddington/(eta_interpolated*(c**2))
        spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star)
        log_vs = np.log10(vs)
        tot = trapezoid(spec, x=vs)
        totals.append(tot)
        specs.append(spec)
        print(i)
    
    return totals, specs, log_vs

#Convergence testing on one of the spectra from the above function
def convergence_test_one_spec_edd(r_out, v_start, v_fin, M, v_bins, ref_bins, a_star):
    start = time.time()
    bins_test = [500, 1000, 2000, 5000, 10000]

    if a_star >= 0:
        r_in = r_ms_prograde(a_star)
    else:
        r_in = r_ms_retrograde(a_star)
    print(len(a_stars))
    print(len(efficiency))
    eta_interpolated = np.interp(a_star, a_stars, efficiency)
    M_dot = L_edd(M)/(eta_interpolated*(c)**2)
    ref_spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, ref_bins, a_star)
    log_vs = np.log10(vs)
    ref_tot = trapezoid(ref_spec, x=vs)
    all_spectrums = np.array(ref_spec)
    for bins in bins_test:
        spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star) 
        all_spectrums = np.vstack((all_spectrums, spec))
    normalised_spectrums = all_spectrums/all_spectrums[0, :]

    counter = -1
    for row in normalised_spectrums:
        if counter == -1:
            plt.plot(log_vs, row, label = f"Reference Spectrum - {ref_bins} bins")
        else:
            plt.plot(log_vs, row, label = f"{bins_test[counter]} bins")
        counter += 1

    plt.xlabel('$log_{10}$($\\nu$ / Hz)', fontsize = 16)
    plt.ylabel('$\\frac{L_{\\nu}}{L_{\\nu}(ref)}$', fontsize = 16)
    # plt.title(f'Convergence testing for spectrum, M={M}$M_{{\odot}}$, \u03B7={eta}')
    plt.legend(loc = 'best')

    end = time.time()
    print(end - start)

    return plt.show()
    
#Convergence testing on all spectra from the spectrum_edd_vary_a_star function.
#This also does convergence testing on L_Tot
def convergence_test_spec_edd(r_out, v_start, v_fin, M, v_bins, bins, a_stars):
    start = time.time()
    bins_test = [500, 1000, 2000, 5000, 20000]
    log_bins = np.log10(bins_test)
    
    for a_star in a_stars:
        if a_star >= 0:
            r_in = r_ms_prograde(a_star)
        else:
            r_in = r_ms_retrograde(a_star)
        eta_interpolated = np.interp(a_star, a_stars, efficiency)
        M_dot = L_edd(M)/(eta_interpolated*(c)**2)
        ref_spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, ref_bins, a_star)
        log_vs = np.log10(vs)
        ref_tot = trapezoid(ref_spec, x=vs)
        total_ratios = []
        all_spectrums = np.array(ref_spec)
        for bins in bins_test:
            spec, vs, log_vl_v = spectrum(r_in, r_out, v_start, v_fin, M, M_dot, v_bins, bins, a_star) 
            tot = trapezoid(spec, x=vs)
            total_ratios.append(tot/ref_tot)
            all_spectrums = np.vstack((all_spectrums, spec))
        normalised_spectrums = all_spectrums/all_spectrums[0, :]
        plt.figure()
        counter = -1
        for row in normalised_spectrums:
            if counter == -1:
                plt.plot(log_vs, row, label = f"Reference Spectrum - {ref_bins} bins")
            else:
                plt.plot(log_vs, row, label = f"{bins_test[counter]} bins")
            counter += 1

        plt.xlabel('$log_{10}$($\\nu$ / Hz)')
        plt.ylabel('$\\frac{L_{\\nu}}{L_{\\nu}(ref)}$')
        plt.title(f'Convergence testing for spectrum, M={M}$M_{{\odot}}$, \u03B7={eta}')
        plt.legend(loc = 'best')
    
        plt.figure()
        plt.plot(log_bins, total_ratios)
        plt.plot(log_bins, [1,1,1,1,1]) #This represents the reference spectrum of 10000 bins
        plt.xlabel('$log_{10}$(No of bins)')
        plt.ylabel('$\\frac{L_{Tot}}{L_{Tot}(ref)}$')
        plt.title(f'Convergence testing for $L_{{Tot}}$, M={M}$M_{{\odot}}$, \u03B7={eta}')

    end = time.time()
    print(end - start)

    return plt.show()