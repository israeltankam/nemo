#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Set page layout to centered and responsive
# st.set_page_config(layout="centered")
# Set default values of parameters
# Define default parameter values
st.session_state.setdefault("a_freq", 0.05)
st.session_state.setdefault("init_infest", 80.0)
st.session_state.setdefault("deployment_type", "R")
st.session_state.setdefault("rep_deployment", True)
st.session_state.setdefault("num_generations", 20)
st.session_state.setdefault("bc", 0.0)
st.session_state.setdefault("sav", 0.25)
st.session_state.setdefault("sv", 0.25)
st.session_state.setdefault("rs", 0.35)
st.session_state.setdefault("s_E", 0.036)
st.session_state.setdefault("N", 300)
st.session_state.setdefault("detection_threshold", 0.1)
step = 0.01
h = 0.88                       
rr = 1                              
st.session_state.M = 250*st.session_state.N
s = 1
v = [1] * st.session_state.num_generations

# Set Streamlit app title
st.title("Nemo")
def dec2(x):
    d = round(x * 100) / 100
    return d
def ratio(u,rs,rr):
    r_attrib = []
    for val in u:
        if val == 0:
            r_attrib.append(rs)
        else:
            r_attrib.append(rr)
    return r_attrib

def survival(u,sv,sav):
    s_attrib = []
    for val in u:
        if val == 0:
            s_attrib.append(sv)
        else:
            s_attrib.append(sav)
    return s_attrib

def attrib_constants(u,rs,rr,sv,sav):
    M_A = [s * r for s, r in zip(survival(u,sv,sav), ratio(u,rs,rr))]
    M_a = [sv * rs] * len(u)
    F_A = [(s * (1 - r)) for s, r in zip(survival(u,sv,sav), ratio(u,rs,rr))]
    F_a = [sv * (1 - rs)] * len(u)
    return M_A, M_a, F_A, F_a

def generate_deployment_vector(input_string, n_gen):   #n_gen will be the global st.session_state.num_generations
    if input_string == 'R':
        return [1] * n_gen
    elif input_string == 'S':
        return [0] * n_gen
    else:
        vector = [1 if char == 'R' else 0 for char in input_string]
        return vector
      
def generate_main_plot(tot,f_A, f_a, Y, Z):
    fig, ax = plt.subplots(figsize=(18, 14), dpi=100)
    ax.plot(np.arange(1, st.session_state.num_generations+1), np.divide(tot,st.session_state.N), '-r', linewidth=3)
    ax.set_xlabel("Generations", fontsize=40)
    ax.set_ylabel("Cysts/g of soil (log)", fontsize=40)
    ax.set_xlim([1, st.session_state.num_generations])
    ax.set_ylim([st.session_state.detection_threshold, st.session_state.M/st.session_state.N])
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # Create two columns with widths in the ratio 2:1
    col1, col2 = st.columns([2, 1])
    with col1:
        # Display the plot in Streamlit
        st.pyplot(fig)
    with col2:
        # Upper plot
        with st.expander("Frequency of avirulence allele A"):
            # Create a new figure and axes
            fig_upper, ax_upper = plt.subplots(figsize=(10, 8), dpi=100)

            # Plot the upper plot data
            ax_upper.plot(np.arange(1, st.session_state.num_generations+1), f_A, linewidth=3)
            ax_upper.set_xlabel("Generations", fontsize=40)
            #ax_upper.set_ylabel("Frequency of allele A")
            #ax_upper.set_title("Frequency of allele avirulence A", fontsize=40)
            ax_upper.tick_params(axis='both', which='major', labelsize=30)

            # Display the upper plot using Streamlit's pyplot function
            st.pyplot(fig_upper)

        # Lower plot
        with st.expander("Frequency of virulence allele a"):
            # Create a new figure and axes
            fig_lower, ax_lower = plt.subplots(figsize=(10, 8), dpi=100)

            # Plot the lower plot data
            ax_lower.plot(np.arange(1, st.session_state.num_generations+1), f_a, linewidth=3)
            ax_lower.set_xlabel("Generations", fontsize=40)
            #ax_lower.set_ylabel("Frequency of allele a")
            #ax_lower.set_title("Frequency of allele virulence a", fontsize=40)
            ax_lower.tick_params(axis='both', which='major', labelsize=30)
            # Display the lower plot using Streamlit's pyplot function
            st.pyplot(fig_lower)


# Main tab
with st.sidebar:
    main_tab = st.radio("Navigation", ["Introduction", "Model & Parameters", "Simulation", "Settings"])

if main_tab == "Introduction":
    st.markdown("# Introduction")
    st.markdown("- Globodera pallida, or Potato Cyst Nematode (PCN), is a serious quarantine pest that threatens potato crops worldwide.")
    st.markdown("- The use of resistant potato cultivars is a popular sustainable pest control measure, but the evolution of PCN populations towards virulence can reduce the long-term effectiveness of resistance-based control.")
    st.markdown("- Masculinizing resistance prevents avirulent nematodes from producing females, which could ultimately eliminate avirulent PCNs from the population.")
    st.markdown("- However, [Shouten's model](https://link.springer.com/article/10.1007/BF03041409) tracing genotypic frequencies in real conditions shows that The long-term fixation of the virulence allele does not occur despite the selection pressure.")
    st.markdown("- Avirulent nematodes, which are exclusively male, survive as heterozygotes by mating with virulent females, weakening the PCN's reproductive rate.")
    st.markdown("- Biocontrol efficiency required for PCN long-term suppression under resistant plants is lower than under susceptible plants.")
    st.markdown("- Combining resistant cultivars with biocontrol methods appears to be an effective solution for suppressing PCN populations.")
    st.markdown("- The model presented for this simulation tracks at the same time the PCN genetics and dynamics to describe selection for virulence and biocontrol needs under resistance.")
    st.markdown("- The user is able to enter the type of plant deployed in each generation (season) - S for Susceptible, R for Resistant - and the app will establish PCN's base reproductive rate, its threshold for nematode suppression , and the evolution of the PCN population as well as the corresponding allele frequencies.")

elif main_tab == "Model & Parameters":
    st.markdown("# Model & parameters")
    image1_path = "figs/diagram.png"
    st.image(image1_path)
    checkbox = st.checkbox("See the simple models")
    if checkbox:
        st.markdown("$X_n \longrightarrow AA$ PCNs, $\qquad Y_n \longrightarrow Aa$ PCNs, $\qquad Z_n \longrightarrow aa$ PCNs")
        st.markdown("- When susceptible plants are deployed in every generation, the overall PCN population $N_n$ at generation $n$ is given by the law:")
        st.latex(r'''
        \begin{equation*}
            N_{n+1}  = \frac{R_0KN_n}{K + N_n}, 
        \end{equation*}
        ''')
        st.markdown("Where $R_0$ is PCN basic reproduction number and K is a population limiting fator due to competition. The different genotype frequencies are given by the Hardy-Weinberg principle.")
        st.markdown("- When resistance plants are deployed in every generation, the overall PCN population $N_n$ at generation $n$ is given by the law:")
        st.latex(r'''
        \begin{equation*}
                \left\{\begin{aligned}
                    N_{n+1} &= \displaystyle\frac{R_0K}{K+N_n}N_n(2a_n-1) \\
                    a_{n+1} &= \frac{1}{2}\Bigg[\displaystyle\frac{2M_a(2a_n-1) + 3 M_A(1-a_n)}{M_a(2a_n-1) + 2M_A(1-a_n)}\Bigg]
                \end{aligned}\right.
        \end{equation*}
        ''')
        st.markdown("Where $N_n$ tracks the population dynamics while $a_n$ tracks the frequency of allele $a_n$. $M_A$ (resp. $M_a$) is the proportion larvae that becomes avirulent (resp. virulent) male adults")
    st.markdown("### Basic reproduction number")
    checkbox = st.checkbox("Read the text")
    if checkbox:
        st.markdown("- The basic reproduction number $R_0$ is the product of the the survival rate $s_v$ of virulent nematodes, the female allocation on susceptible plants $(1-r)$ and the PCN growth factor $G$.")
        st.latex(r'''
        \begin{equation*}R_0 = Gs_v(1-r)
        \end{equation*}
        ''')
        st.markdown("- The growth factor G is the product of hatching success ($h$), average number of larvae per cyst ($k$), survival of encysted larvae ($s_E$).")
        st.latex(r'''
        \begin{equation*}G = hkS_E
        \end{equation*}
        ''')
        markdown_text = r'''
        Blocking resistances disappear completely over time provided that the resistance gene is fixed by natural selection. Thus, as with susceptible plants, the long-term suppression of PCNs must be ensured by a biocontrol that brings the basic reproduction number $\mathcal{R}_0$ of PCNs below 1.

        On the other hand, masculinizing resistances keep a partial resistance indefinitely. This is conditioned by a parameter $\alpha = \frac{M_a}{M_A}$ which is the ratio between the larvae which produce male adults and those which produce female adults. When $\alpha < \frac{1}{2}$, which is the case in reality, it suffices to ensure the long-term suppression of PCN that the biocontrol brings $\mathcal{R}_0$ below $2(1-\alpha)$ as shown in the figure below. Partial resistance is ensured by the survival of susceptible phenotype through the pairing of avirulent males with virulent females.
        '''

        st.markdown(markdown_text)
    image2_path = "figs/diag_r0_marked.png"
    st.image(image2_path)
    st.markdown("### Parameters")
    table_md = r'''
    | Parameter | Description | Value | Range |
    | --- | --- | --- | --- |
    | $s_v$ | Survival rate of avirulent PCNs on resistant plants | $25\%$ | [0,100\%] |
    | $s_{av}$ | Survival rate of virulent PCNs on resistant plants | $25\%$ | [0,100\%] |
    | $h$ | Cyst hatching success rate | $88\%$ | [60\%,97\%] |
    | $r$ | Average male allocation on susceptible plants | $35\%$ | (0, 35\%] |
    | $s_E$ | Survival rate of encysted juveniles | $3.6\%$ | [0.01\%, 10\%] |
    | $k$ | Average number of eggs per cyst | 300 | [200, 500] |
    | $\beta_c$ | Efficacy of the biocontrol | variable | [0, 99.9\%] |
    | $K$ | PCN limiting factor | 250 cysts/g of soil | -- |
    |  | Detection (cleaning) threshold | 0.1 cyst/g of soil | [0.1, 1] cyst/ g of soil |
    '''
    st.markdown(table_md)
    st.markdown("In this table as in the simulation, the **efficacy of biocontrol** is modulable and we can analyze its effect on the PCN reproduction number and suppression. Other modulable parameters for simulation are the **initial frequency of virulence allele** and the **initial soil infestation**.")
    st.markdown("The detection threshold defines the PCN level below which the field is considered clean")
    st.markdown("The simulation allows to choose the plant breed which is deployed per season. This defines the PCN suppression threshold. The other parameters, very little variable and estimated on literature data, are in the Settings menu.")
    # Create a checkbox to toggle the hidden content
    checkbox = st.checkbox("See the general model")
    if checkbox:
        st.markdown("- More generally, under arbitrary deployment of susceptible and resistant plants, the model reads:")
        st.latex(r'''
        \begin{equation*}
                \left\{\begin{aligned}
                X_{n+1} &= \frac{G K}{K + (X_n+Y_n+Z_n)} \displaystyle\frac{M_A(n)F_A(n)\big(X_n + \frac{1}{2}Y_n\big)^2}{M_A(n)(X_n+Y_n) + M_a(n)Z_n}, \\
                \\
                Y_{n+1} &=  \frac{G K}{K + (X_n+Y_n+Z_n)}  \displaystyle\frac{M_A(n)\big(F_a(n)Z_n + \frac{1}{2}F_A(n)Y_n \big)\big(X_n + \frac{1}{2}Y_n \big) + F_A(n)\big(X_n + \frac{1}{2}Y_n \big)\big(M_a(n)Z_n + \frac{1}{2}M_A(n)Y_n \big) }{M_A(n)(X_n+Y_n) + M_a(n)Z_n},\\
                \\
                Z_{n+1} &= \frac{G K}{K + (X_n+Y_n+Z_n)} \displaystyle\frac{\big(F_a(n)Z_n + \frac{1}{2}F_A(n)Y_n\big)\big(M_a(n)Z_n + \frac{1}{2}M_A(n)Y_n\big)}{M_A(n)(X_n+Y_n) + M_a(n)Z_n},
            \end{aligned}\right.
        \end{equation*}
        ''')
        st.markdown("Where $F_A$ (resp. $F_a$) is the proportion larvae that becomes avirulent (resp. virulent) female adults")
    # Add a link to expand/collapse the hidden content
    #st.markdown("[Expand / Collapse](javascript:void(0);)")
    
    
elif main_tab == "Simulation":
    st.markdown("# Simulation")
    # Other parameters
    step = 0.01
    h = 0.88                           #Egg hatching success rate
    rr = 1                             #Avirulent pest sex ratio when resistance
    st.session_state.M = 145*st.session_state.N  #Limiting factor
    v = generate_deployment_vector(st.session_state.deployment_type, st.session_state.num_generations)
    s = 1
    g = h*st.session_state.N*st.session_state.s_E*s*(1-st.session_state.bc)
    alpha = st.session_state.sv*st.session_state.rs/st.session_state.sav
    R0 = g*st.session_state.sv*(1-st.session_state.rs)
    if np.all(v == np.ones(st.session_state.num_generations)):
        thres = max(dec2(2 * (1 - alpha)),1)
    elif np.all(v == np.zeros(st.session_state.num_generations)):
        thres = 1
    else:
        thres = float('-inf')
    init_larvae = st.session_state.init_infest*st.session_state.N*h #Cysts x nbre of eggs per cyst * hatching success * converstion to 1ha field * field st.session_state.area
    J_AA_0 = (1-st.session_state.a_freq)**2*init_larvae
    J_Aa_0 = 2*st.session_state.a_freq*(1-st.session_state.a_freq)*init_larvae
    J_aa_0 = (st.session_state.a_freq)**2*init_larvae ##Hardy-Weinberg
    # Generate plots based on the input parameters
    # ...
    X = np.zeros(st.session_state.num_generations)
    Y = np.zeros(st.session_state.num_generations)
    Z = np.zeros(st.session_state.num_generations)
    X[0] = J_AA_0
    Y[0] = J_Aa_0
    Z[0] = J_aa_0
    M_A, M_a, F_A, F_a = attrib_constants(v,st.session_state.rs,rr,st.session_state.sv,st.session_state.sav)
    for n in range(st.session_state.num_generations-1):
        if X[n] + Y[n] + Z[n] == 0:
            X[n+1] = 0
            Y[n+1] = 0
            Z[n+1] = 0
        else:
            X[n+1] = (g * (M_A[n] * F_A[n] * (X[n] + Y[n]/2)**2) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n])) / (1 + (X[n] + Y[n] + Z[n]) / st.session_state.M)
            Y[n+1] = (g * (M_A[n] * (F_a[n] * Z[n] + F_A[n] * Y[n]/2) * (X[n] + Y[n]/2) + F_A[n] * (M_a[n] * Z[n] + M_A[n] * Y[n]/2) * (X[n] + Y[n]/2))) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n]) / (1 + (X[n] + Y[n] + Z[n]) / st.session_state.M)
            Z[n+1] = (g * (F_a[n] * Z[n] + F_A[n] * Y[n]/2) * (M_a[n] * Z[n] + M_A[n] * Y[n]/2)) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n]) / (1 + (X[n] + Y[n] + Z[n]) / st.session_state.M)
    
    tot = X + Y + Z
    f_AA = np.zeros(st.session_state.num_generations)
    f_Aa = np.zeros(st.session_state.num_generations)
    f_aa = np.zeros(st.session_state.num_generations)
    f_A = np.zeros(st.session_state.num_generations)
    f_a = np.zeros(st.session_state.num_generations)

    for n in range(st.session_state.num_generations):
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
    # Display R0
    R0 = dec2(R0)
    disp = thres if thres != float('-inf') else 'Unknown'
    cl1, cl2 = st.columns([1, 3])
    cl1.markdown("")  # Placeholder for the first empty column
    cl2.markdown(f"$R_0$ = {R0}")
    cl2.markdown(f"Suppression threshold = {disp}")
    generate_main_plot(tot,f_A, f_a, Y, Z)
    st.session_state.a_freq = st.slider("Frequency of the virulence allele (%):", min_value=0.0, max_value=99.9, value=st.session_state.a_freq*100, step=0.1)/100
    st.session_state.init_infest = st.slider("Initial infestation (cysts/g of soil):", min_value=0.1, max_value=240.0, value=st.session_state.init_infest, step=0.1)
    st.session_state.deployment_type = st.text_input("Deployment type:", value=st.session_state.deployment_type)
    st.session_state.deployment_type = st.session_state.deployment_type.upper()  # Convert input to uppercase
    if not all(ch in ['R', 'S'] for ch in st.session_state.deployment_type):
        st.error("Invalid deployment type. Please enter a string containing only 'R' or 'S'.")
    #update_button = st.button("Update")
    st.session_state.rep_deployment = st.checkbox("Indefinitely repeat this deployment", value=True)
    if st.session_state.rep_deployment:
        st.session_state.num_generations = st.slider("Number of generations (if checked above):", min_value=1, max_value=100, value=st.session_state.num_generations, step=1)
    else:
        st.session_state.num_generations = len(st.session_state.deployment_type)
    st.session_state.bc = st.slider("Efficacy of biocontrol (%):", min_value=0.0, max_value=99.9, value=st.session_state.bc*100, step=0.1)/100
    st.session_state.detection_threshold = st.slider("Detection threshold (cysts/g of soil):", min_value=0.01, max_value=1.0, value=st.session_state.detection_threshold, step=0.01)
    if not st.session_state.rep_deployment:
        st.session_state.num_generations = len(st.session_state.deployment_type)
elif main_tab == "User guide":
    st.markdown("## User guide")
    # Edit and add your user guide text here

elif main_tab == "Settings":
    st.markdown("# Settings")
    st.markdown("These parameters describe the basic biology of PCNs. They are retrieved from intensive literature review and cautious estimations. Please edit these settings if and only if you have enough knowledge!!")
    st.session_state.sav = st.slider("Survival of avirulent PCNs on resistant plants (%):", min_value=0.0, max_value=100.0, value=st.session_state.sav*100, step=0.1)/100
    st.session_state.sv = st.slider("Survival of virulent PCNs on resistant plants (%):", min_value=0.0, max_value=100.0, value=st.session_state.sv*100, step=0.1)/100
    st.session_state.rs = st.slider("Average male allocation on susceptible potato (%):", min_value=0.0, max_value=100.0, value=st.session_state.rs*100, step=0.1)/100
    st.session_state.s_E = st.slider("Survival of larvae from cysts (%):", min_value=0.010, max_value=10.0, value=st.session_state.s_E*100, step=0.001)/100
    st.markdown("(Includes suicidal hatching and larval desiccation)")
    st.session_state.N = st.slider("Average eggs per cyst:", min_value=200, max_value=500, value=st.session_state.N, step=1)


# In[ ]:




