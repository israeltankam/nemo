#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import hydralit_components as hc
import matplotlib.pyplot as plt
import numpy as np

# Set page layout to centered and responsive
# st.set_page_config(layout="wide")
st.set_page_config(layout='wide',initial_sidebar_state='collapsed')


# specify the primary menu definition
menu_data = [
    {'icon': "far fa-copy", 'label':"Model & Parameters"},
    {'icon': "far fa-chart-bar", 'label':"Simulation"},#no tooltip message
    {'icon': "fas fa-tachometer-alt", 'label':"Settings"},
]

over_theme = {'txc_inactive': '#FFFFFF', 'menu_background':'#85929E'}
st.markdown("# Nemo")
main_tab= hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Introduction',
    #login_name='Logout',
    hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)


# Define default parameter values
st.session_state.setdefault("a_freq", 0.0167)
st.session_state.setdefault("init_infest_cyst", 0.5)
st.session_state.setdefault("deployment_type", "RNNNNRNNNNRNNNNRNNNNR")
st.session_state.setdefault("bc", 0.0)
st.session_state.setdefault("s", 0.25)
st.session_state.setdefault("m", 0.35)
st.session_state.setdefault("ha", 0.17)
st.session_state.setdefault("v", 0.90)
st.session_state.setdefault("M", 15.9)
st.session_state.setdefault("e", 300)
st.session_state.setdefault("detection_threshold", -3)
step = 0.01
                            

# Define parameter values for reset
st.session_state.setdefault("reset_a_freq", 0.0167)
st.session_state.setdefault("reset_init_infest_cyst", 0.5)
st.session_state.setdefault("reset_deployment_type", "RNNNNRNNNNRNNNNRNNNNR")
st.session_state.setdefault("reset_bc", 0.0)
st.session_state.setdefault("reset_s", 0.25)
st.session_state.setdefault("reset_m", 0.35)
st.session_state.setdefault("reset_ha", 0.17)
st.session_state.setdefault("reset_v", 0.90)
st.session_state.setdefault("reset_M", 15.9)
st.session_state.setdefault("reset_e", 300)
st.session_state.setdefault("reset_detection_threshold", -3)

# Set Streamlit app title
#st.title("Nemo")
def dec2(x):
    d = round(x * 100) / 100
    return d
def ratio(u,r):
    r_attrib = []
    for val in u:
        if val == 0:
            r_attrib.append(r)
        else:
            r_attrib.append(1)
    return r_attrib

def attrib_constants(u,r):
    M_A = [st.session_state.s * r_attributed for r_attributed in ratio(u,r)]
    M_a = [st.session_state.s * r] * len(u)
    F_A = [(st.session_state.s * (1 - r_attributed)) for r_attributed in ratio(u,r)]
    F_a = [st.session_state.s * (1 - r)] * len(u)
    return M_A, M_a, F_A, F_a

def generate_deployment_vector(input_string):
    temp = ""
    n_count = 0

    for char in input_string:
        if char == "N":
            n_count += 1
        else:
            if n_count > 0:
                temp += str(n_count)
                n_count = 0

            if temp and (temp[-1] == char or (temp[-1] == "S" and char == "R") or (temp[-1] == "R" and char == "S")):
                temp += "0"
            temp += char

    if n_count > 0:
        temp += str(n_count)

    deployment = ""
    jn = []

    num_buffer = ""
    for char in temp:
        if char.isdigit():
            num_buffer += char
        else:
            if num_buffer:
                jn.append(int(num_buffer))
                num_buffer = ""
            deployment += char

    if num_buffer:
        jn.append(int(num_buffer))

    jn_vector = [int(num) for num in jn]  # Convert jn to int vector
    deployment_vector = [1 if char == 'R' else 0 for char in deployment]
    if len(jn_vector)==len(deployment_vector):
        jn_vector = jn_vector[:-1] # To discard deployment ended by Non-Host
    return deployment_vector, jn_vector
      
def generate_main_plot(tot,f_A, f_a, Y, Z,R):
    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
    nb_gen = len(tot)
    ax.plot(np.arange(1, nb_gen+1), tot, '-r', linewidth=3)
    th = st.session_state.e*10**(st.session_state.detection_threshold)
    ax.plot([0, nb_gen], [th, th], 'k--', label='Detection threshold')
    ax.set_xlabel("Generations", fontsize=40)
    ax.set_ylabel("PCNs/g of soil (log)", fontsize=40)
    ax.set_xlim([1, nb_gen])
    ax.set_ylim([10**(-6), st.session_state.M*(R-1)])
    ax.set_yscale('log')
    tick_locations = [10**(-6), 10**0, th, 10**1, 10**2, 10**3]
    tick_labels = [str(val) for val in [0] + tick_locations[1:]]
    ax.set_yticks(tick_locations, tick_labels)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.legend(fontsize=15)
    # Create two columns with widths in the ratio 2:1
    col1, col2 = st.columns([2, 1])
    with col1:
        # Display the plot in Streamlit
        st.pyplot(fig)
    with col2:
        # Upper plot
        with st.expander("Frequency of avirulence allele A"):
            # Create a new figure and axes
            fig_upper, ax_upper = plt.subplots(figsize=(8, 5), dpi=100)

            # Plot the upper plot data
            ax_upper.plot(np.arange(1, nb_gen+1), f_A, linewidth=3)
            ax_upper.set_xlabel("Generations", fontsize=40)
            #ax_upper.set_ylabel("Frequency of allele A")
            #ax_upper.set_title("Frequency of allele avirulence A", fontsize=40)
            ax_upper.tick_params(axis='both', which='major', labelsize=30)

            # Display the upper plot using Streamlit's pyplot function
            st.pyplot(fig_upper)

        # Lower plot
        with st.expander("Frequency of virulence allele a"):
            # Create a new figure and axes
            fig_lower, ax_lower = plt.subplots(figsize=(8, 5), dpi=100)

            # Plot the lower plot data
            ax_lower.plot(np.arange(1, nb_gen+1), f_a, linewidth=3)
            ax_lower.set_xlabel("Generations", fontsize=40)
            #ax_lower.set_ylabel("Frequency of allele a")
            #ax_lower.set_title("Frequency of allele virulence a", fontsize=40)
            ax_lower.tick_params(axis='both', which='major', labelsize=30)
            # Display the lower plot using Streamlit's pyplot function
            st.pyplot(fig_lower)



# Main tab
#with st.sidebar:
#    main_tab = st.radio("Navigation", ["Introduction", "Model & Parameters", "Simulation", "Settings"])

if main_tab == "Introduction":
    st.markdown("# Introduction")
    st.markdown("- Globodera pallida, or Pale Cyst Nematode (PCN), is a serious quarantine pest that threatens potato crops worldwide.")
    st.markdown("- The use of resistant potato cultivars is a popular sustainable pest control measure, but the evolution of PCN populations towards virulence can reduce the long-term effectiveness of resistance-based control.")
    st.markdown("- Masculinizing resistance prevents avirulent nematodes from producing females, which could ultimately eliminate avirulent PCNs from the population.")
    st.markdown("- However, [Shouten's model](https://link.springer.com/article/10.1007/BF03041409) tracing genotypic frequencies in real conditions shows that the long-term fixation of the virulence allele does not necessarily occur despite the selection pressure.")
    st.markdown("- Avirulent nematodes, which are exclusively male, survive as heterozygotes by mating with virulent females, weakening the PCN's reproduction number.")
    st.markdown("- Biocontrol efficiency required for PCN long-term suppression under resistant plants is lower than under susceptible plants.")
    st.markdown("- But the efficiency required even under resistant plant can be very high, thus unachievable")
    st.markdown("- Rotations are proven to be an efficient and sustainable lever of PCN control, but required long periods of cultivating non-host crops or leaving a bare soil")
    st.markdown("- Combining resistant cultivars with biocontrol methods and rotations appears to be an effective solution for speeding up the suppression of PCN populations.")
    st.markdown("- The model presented for this simulation tracks at the same time the PCN genetics and dynamics to describe selection for virulence and biocontrol+rotation size needs under resistance.")
    st.markdown("- The user is able to enter the type of crop for each season - S for Susceptible, R for Resistant, N for Non-host (corresponding to a rotation) - and the app will draw the PCN population trajectories as well as the corresponding allele frequencies.")

elif main_tab == "Model & Parameters":
    st.markdown("# Model & parameters")
    image1_path = "figs/diagram.png"
    st.image(image1_path)
    checkbox = st.checkbox("See the simple models")
    if checkbox:
        st.markdown("$X_n \longrightarrow AA$ PCNs, $\qquad Y_n \longrightarrow Aa$ PCNs, $\qquad Z_n \longrightarrow aa$ PCNs")
        st.markdown("- When susceptible plants are deployed in every generation and $j$-years rotations are always implemented between generations, the overall PCN population $N_k$ at generation $k$ is given by the law:")
        st.latex(r'''
        \begin{equation*}
            N_{k+1} = (1-m)R_j \displaystyle\frac{K N_k}{K+ N_k\big( (1-m)R_j - 1 \big)} 
        \end{equation*}
        ''')
        st.markdown("Where $R_j$ is the PCN reproduction number under $j$-year rotations (corresponding to a basic reproduction number $(1-m)R_j$) and K is the nematode carrying capacity. The different genotype frequencies are given by the Hardy-Weinberg principle.")
        st.markdown("- When resistance plants are deployed in every generation, the overall PCN population $N_k$ at generation $k$ is given by the law:")
        st.latex(r'''
        \begin{equation*}
            \left\{\begin{aligned}
                N_{k+1} &= (1-m)R_j \displaystyle\frac{K N_k}{K+ N_k\big( (1-m)R - 1 \big)}v_k\\
                v_{k+1} &= \displaystyle\frac{m v_k + \frac{1}{2}(1-v_k)}{m v_k + (1-v_k)}
        \end{aligned}\right.
        \end{equation*}
        ''')
        st.markdown("Where $N_k$ tracks the population dynamics while $v_k$ tracks the frequency of virulent nematodes (aa) and $m$ represents the relative proportion of juveniles that develop into virulent males to those that develop into avirulent males")
    st.markdown("### Basic reproduction number")
    checkbox = st.checkbox("Read the text")
    if checkbox:
        st.markdown("- When regular $j$-years rotations are implemented along with a biocontrol of efficacy $b$, the $R_j$ reproduction number takes the following form.")
        st.latex(r'''
        \begin{equation*}R_j = es[w(1-h_a)(1-b)]^{j+1},
        \end{equation*}
        ''')
        st.markdown("Where s is the proportion of larvae that survive the larval stage to become adults.")
        
        markdown_text = r'''
        
        
        Blocking resistances quickly become obsolete as the resistance gene is fixed by natural selection. Thus, as with susceptible plants, the long-term suppression of PCNs must be ensured by a biocontrol that brings the nematode reprodution number $R_j$ below 1/(1-m).

        On the other hand, masculinizing resistances keep a partial resistance indefinitely. This is conditioned by the male allocation rate $m$. When $m < \frac{1}{2}$, which is the case in real setups, it suffices to ensure the long-term suppression of PCNs
        that control efforts bring the reproduction number $R_j$ below $2$. The partial resistance is ensured by the survival
        of susceptible phenotype through the pairing of avirulent males with virulent females.
        '''

        st.markdown(markdown_text)
    image2_path = "figs/scenario_diagram.png"
    st.image(image2_path, width=1000)
    st.markdown("### Parameters")
    table_md = r'''
    | Parameter | Description | Value | Range |
    | --- | --- | --- | --- |
    | $s$ | Survival rate of hatched larvae | $25\%$ | [0,100\%] |
    | $m$ | Male allocation on susceptible plants | $35\%$ | (0, 35\%] |
    | $e$ | Average number of eggs per cyst | 300 | [200, 500] |
    | $w$ | Yearly viability of encysted larvae | $90\%$ | [80, 99.9\%] |
    | $h_a$ | Yearly rate of accidental hatching | 17% | [0, 35\%] |
    | $b$ | Efficacy of the biocontrol | variable | [0, 99.9\%] |
    | $K$ | Nematode carrying capacity| 170 g$^{-1}$ soil | -- |
    |  | Detection (cleaning) threshold | 0.01 cyst/g of soil | [0.01, 0.1] cyst/ g of soil |
    '''
    st.markdown(table_md)
    #st.markdown("In this table as in the simulation, the **efficacy of biocontrol** is modulable and we can analyze its effect on the PCN reproduction number and suppression. Other modulable parameters for simulation are the **initial frequency of virulence allele** and the **initial soil infestation**.")
    #st.markdown("The detection threshold defines the PCN level below which the field is considered clean")
    st.markdown("The simulation allows to choose the plant breed which is deployed per season. The other parameters, very little variable and estimated on literature data, are in the Settings menu.")
    # Create a checkbox to toggle the hidden content
    checkbox = st.checkbox("See the general model")
    if checkbox:
        st.markdown("- More generally, under arbitrary deployment of susceptible and resistant plants, the model reads:")
        st.latex(r'''
        \begin{equation*}
            \left\{\begin{aligned}
            X_{k+1} &= \frac{G_{j_k} M}{M + (X_k+Y_k+Z_k)} \displaystyle\frac{M_A(k)F_A(k)\big(X_k + \frac{1}{2}Y_k\big)^2}{M_A(k)(X_k+Y_k) + M_a(k)Z_k}, \\
        \\
        Y_{k+1} &=  \frac{G_{j_k} M}{M + (X_k+Y_k+Z_k)}  \displaystyle\frac{M_A(k)\big(F_a(k)Z_k + \frac{1}{2}F_A(k)Y_k \big)\big(X_k + \frac{1}{2}Y_k \big) + F_A(k)\big(X_k + \frac{1}{2}Y_k \big)\big(M_a(k)Z_k + \frac{1}{2}M_A(k)Y_k \big) }{M_A(k)(X_k+Y_k) + M_a(k)Z_k},\\
        \\
        Z_{k+1} &= \frac{G_{j_k} M}{M + (X_k+Y_k+Z_k)} \displaystyle\frac{\big(F_a(k)Z_k + \frac{1}{2}F_A(k)Y_k\big)\big(M_a(k)Z_k + \frac{1}{2}M_A(k)Y_k\big)}{M_A(k)(X_k+Y_k) + M_a(k)Z_k},
        \end{aligned}\right.
    \end{equation*}
        ''')
        st.markdown("Where $F_A$ (resp. $F_a$) is the proportion larvae that becomes avirulent (resp. virulent) female adults, $G_{j_k}$ is the generation $k$'s growth factor ($G_{j_k} = e[w(1-h_a)(1-b)]^{j_k+1}$ ) provided there is are $j_k$-year rotations after generation $k$, and $M$ is a limiting factor.")
    # Add a link to expand/collapse the hidden content
    #st.markdown("[Expand / Collapse](javascript:void(0);)")
    
    
elif main_tab == "Simulation":
    st.markdown("# Simulation")
    if st.button("Reset all"):
        st.session_state.a_freq = st.session_state.reset_a_freq
        st.session_state.init_infest_cyst = st.session_state.reset_init_infest_cyst
        st.session_state.deployment_type = st.session_state.reset_deployment_type
        st.session_state.bc = st.session_state.reset_bc
        st.session_state.s = st.session_state.reset_s
        st.session_state.m = st.session_state.reset_m
        st.session_state.ha = st.session_state.reset_ha
        st.session_state.v = st.session_state.reset_v
        st.session_state.M = st.session_state.reset_M
        st.session_state.e = st.session_state.reset_e
        st.session_state.detection_threshold = st.session_state.reset_detection_threshold
    # Other parameters
    colu1, colu2, colu3 = st.columns(3)
    with colu1:
        st.session_state.a_freq = st.slider("Initial frequency of the virulence allele (%):", min_value=0.0, max_value=99.9, value=st.session_state.a_freq*100, step=0.1)/100
        st.markdown("Click to select a type of crop to deploy. Do not begin nor end with Non-host.")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        st.session_state.count_patern = 0
        if 'pattern_temp' not in st.session_state:
            st.session_state.pattern_temp = st.session_state.deployment_type
        
        if col1.button("Susc."):
            st.session_state.deployment_type += "S"
            st.session_state.count_patern = 0
            
        if col2.button("Resis."):
            st.session_state.deployment_type += "R"
            st.session_state.count_patern = 0

        if col3.button("Non-host"):
            if st.session_state.deployment_type != "":
                st.session_state.deployment_type += "N"
                st.session_state.count_patern = 0
                
        if col4.button("Repeat pattern"):
            if st.session_state.count_patern == 0:
                pattern = st.session_state.deployment_type
                st.session_state.count_patern += 1
            else:
                pattern = st.session_state.pattern_temp
                st.session_state.count_patern += 1
            st.session_state.deployment_type += pattern
            

        if col5.button("Delete"):
            if len(st.session_state.deployment_type) > 0:
                st.session_state.deployment_type = st.session_state.deployment_type[:-1]

        if col6.button("Erase"):
            st.session_state.deployment_type = ""

    with colu2:
        st.session_state.init_infest_cyst = st.slider("Initial infestation (cysts/g of soil):", min_value=0.01, max_value=2.0, value=st.session_state.init_infest_cyst, step=0.01)
        # Display the deployment type
        st.markdown("Deployment type")
        st.text(st.session_state.deployment_type)
    with colu3:
        st.session_state.bc = st.slider("Efficacy of biocontrol (%):", min_value=0.0, max_value=99.9, value=st.session_state.bc*100, step=0.1)/100
        st.session_state.detection_threshold = st.slider(f"Detection threshold (10$^\square$ cysts/g of soil)", min_value=-3, max_value=-1, value=int(st.session_state.detection_threshold), step=1)
        
    #u = generate_deployment_vector(st.session_state.deployment_type, st.session_state.num_generations)
    u, jn = generate_deployment_vector(st.session_state.deployment_type)
    #st.markdown(str(u))
    #st.markdown(str(jn))
    if u!=[] and len(u)>len(jn):
        #st.markdown("This is it")
        init_juveniles = st.session_state.init_infest_cyst*st.session_state.e
        J_AA_0 = init_juveniles * (1-st.session_state.a_freq)**2
        J_Aa_0 = init_juveniles * 2 * st.session_state.a_freq*(1-st.session_state.a_freq)
        J_aa_0 = init_juveniles * (st.session_state.a_freq)**2

        # --------------------- thresholds ----------------------------- #
        # ------ jn-growth factors ------------

        Gjn = [st.session_state.e * (st.session_state.v * (1 - st.session_state.ha) * (1 - st.session_state.bc)) ** (j + 1) for j in jn]

        # -------- basic reproduction number ---------
        G0 = st.session_state.e * (st.session_state.v * (1 - st.session_state.ha))
        R0 = G0 * st.session_state.s * (1 - st.session_state.m)

        # ------  alpha -----------------
        alpha = st.session_state.m

        # ------ Effective population number -----------
        # ------ valid if rotation are same-sized -----------
        if all(j == jn[0] for j in jn) and len(jn)>= 1:
            j = jn[0]
            Gj = st.session_state.e * (st.session_state.v * (1 - st.session_state.ha) * (1 - st.session_state.bc)) ** (j + 1)
            Rj = R0 * (Gj / G0)
            #st.markdown("Regular rotation of"); st.markdown(jn[0])
        else:
            Rj = float('inf')
    
        #st.markdown("R0 ="); st.markdown(R0)
        #st.markdown("Resistant threshold ="); st.markdown(2 * (1 - alpha))
        
        nb_gen = len(u)
        X = np.zeros(nb_gen)
        Y = np.zeros(nb_gen)
        Z = np.zeros(nb_gen)
        X[0] = J_AA_0
        Y[0] = J_Aa_0
        Z[0] = J_aa_0
        M_A,M_a,F_A,F_a = attrib_constants(u,st.session_state.m)
        #st.markdown(F_A)
        for n in range(nb_gen-1):
            if X[n] + Y[n] + Z[n] == 0:
                X[n + 1] = 0
                Y[n + 1] = 0
                Z[n + 1] = 0
            else:
                X[n + 1] = (Gjn[n] * (M_A[n] * F_A[n] * (X[n] + Y[n] / 2)**2) /
                            (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n])) / (1 + (X[n] + Y[n] + Z[n]) / st.session_state.M)
                Y[n + 1] = (Gjn[n] * (M_A[n] * (F_a[n] * Z[n] + F_A[n] * Y[n] / 2) * (X[n] + Y[n] / 2) +
                            F_A[n] * (M_a[n] * Z[n] + M_A[n] * Y[n] / 2) * (X[n] + Y[n] / 2)) /
                            (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n])) / (1 + (X[n] + Y[n] + Z[n]) / st.session_state.M)
                Z[n + 1] = (Gjn[n] * (F_a[n] * Z[n] + F_A[n] * Y[n] / 2) * (M_a[n] * Z[n] + M_A[n] * Y[n] / 2) /
                            (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n])) / (1 + (X[n] + Y[n] + Z[n]) / st.session_state.M)

        tot = X + Y + Z
        f_AA = np.zeros(nb_gen)
        f_Aa = np.zeros(nb_gen)
        f_aa = np.zeros(nb_gen)
        f_A = np.zeros(nb_gen)
        f_a = np.zeros(nb_gen)

        for n in range(nb_gen):
            if tot[n] == 0:
                f_AA[n] = 0
                f_Aa[n] = 0
                f_aa[n] = 0
            else:
                f_AA[n] = X[n] / tot[n]
                f_Aa[n] = Y[n] / tot[n]
                f_aa[n] = Z[n] / tot[n]
            
            f_A[n] = f_AA[n] + f_Aa[n] / 2
            f_a[n] = f_aa[n] + f_Aa[n] / 2          
        generate_main_plot(tot,f_A, f_a, Y, Z, R0)
    
    
elif main_tab == "Settings":
    st.markdown("# Settings")
    st.markdown("These parameters describe the basic biology of PCNs. They are retrieved from intensive literature review and cautious estimations. Please edit these settings if and only if you have enough knowledge!!")
    st.session_state.s = st.slider("Survival rate of hatched larvae (%):", min_value=0.0, max_value=100.0, value=st.session_state.s*100, step=0.1)/100
    st.session_state.m = st.slider("Average male allocation on susceptible potato (%):", min_value=0.0, max_value=40.0, value=st.session_state.m*100, step=0.1)/100
    st.session_state.v = st.slider("Viability of encysted juveniles (%):", min_value=80, max_value=100, value=int(st.session_state.v*100), step=1)/100
    st.session_state.ha = st.slider("Yearly rate of accidental hatching (%):", min_value=0, max_value=35, value=int(st.session_state.ha*100), step=1)/100
    st.session_state.e = st.slider("Average eggs per cyst:", min_value=200, max_value=500, value=st.session_state.e, step=1)
