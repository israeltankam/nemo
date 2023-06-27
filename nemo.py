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
st.session_state.setdefault("area", 1.0)
st.session_state.setdefault("a_freq", 0.05)
st.session_state.setdefault("init_infest", 10.0)
st.session_state.setdefault("deployment_type", "R")
st.session_state.setdefault("rep_deployment", True)
st.session_state.setdefault("num_generations", 20)
st.session_state.setdefault("bc", 0.0)
st.session_state.setdefault("sav", 0.25)
st.session_state.setdefault("sv", 0.25)
st.session_state.setdefault("rs", 0.35)
st.session_state.setdefault("s_E", 0.036)
st.session_state.setdefault("N", 350)
step = 0.01
h = 0.88                       
rr = 1                              
conversion_factor = 3.3*10**9
M = 145*st.session_state.N*conversion_factor*st.session_state.area
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
      
def generate_main_plot(tot):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, st.session_state.num_generations+1), tot, '-r')
    ax.set_xlabel("Generations")
    ax.set_ylabel("PCNs in field (log scale)")
    ax.set_xlim([1, st.session_state.num_generations])
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=10)
    # Display the plot in Streamlit
    st.pyplot(fig)


def generate_virulence_plot(f_A, f_a, Y, Z):
    # Frequency of allele A
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(np.arange(1, st.session_state.num_generations+1), f_A)
    ax[0, 0].set_xlabel("Generations")
    ax[0, 0].set_ylabel("Frequency of allele A")
    ax[0, 0].set_xlim([1, st.session_state.num_generations])
    ax[0, 0].set_ylim([0, 1.01])
    ax[0, 0].tick_params(axis='both', which='major', labelsize=10)
    ax[0, 0].legend(['Frequency of allele A'], loc=1, prop={'size': 10})

    # Frequency of allele a
    ax[0, 1].plot(np.arange(1, st.session_state.num_generations+1), f_a)
    ax[0, 1].set_xlabel("Generations")
    ax[0, 1].set_ylabel("Frequency of allele a")
    ax[0, 1].set_xlim([1, st.session_state.num_generations])
    ax[0, 1].set_ylim([0, 1.01])
    ax[0, 1].tick_params(axis='both', which='major', labelsize=10)
    ax[0, 1].legend(['Frequency of allele a'], loc=1, prop={'size': 10})

    # Aa pests
    ax[1, 0].plot(np.arange(1, st.session_state.num_generations+1), Y, '-g')
    ax[1, 0].set_xlabel("Generations")
    ax[1, 0].set_ylabel("Aa PCN")
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_xlim([1, st.session_state.num_generations])
    ax[1, 0].tick_params(axis='both', which='major', labelsize=10)
    ax[1, 0].legend(['Aa PCN'], loc=1, prop={'size': 10})

    # aa pests
    ax[1, 1].plot(np.arange(1, st.session_state.num_generations+1), Z, '-g')
    ax[1, 1].set_xlabel("Generations")
    ax[1, 1].set_ylabel("aa PCN")
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlim([1, st.session_state.num_generations])
    ax[1, 1].tick_params(axis='both', which='major', labelsize=10)
    ax[1, 1].legend(['aa PCN'], loc=1, prop={'size': 10})
    
    
    fig.subplots_adjust(wspace=0.4)  # Adjust the width spacing between subplots
    st.pyplot(fig)

def generate_genetic_drift_plot():
    g = h*st.session_state.N*st.session_state.s_E*s*(1-st.session_state.bc)
    alpha = st.session_state.sv*st.session_state.rs/st.session_state.sav
    R0 = g*st.session_state.sv*(1-st.session_state.rs)
    if np.all(v == np.ones(st.session_state.num_generations)):
        thres = max(dec2(2 * (1 - alpha)),1)
    elif np.all(v == np.zeros(st.session_state.num_generations)):
        thres = 1
    else:
        thres = float('-inf')
    init_larvae = st.session_state.init_infest*st.session_state.N*0.05*h*conversion_factor*st.session_state.area #Cysts x nbre of eggs per cyst * dessication * hatching success * converstion to 1ha field * field st.session_state.area
    J_AA_0 = (1-st.session_state.a_freq)**2*init_larvae
    J_Aa_0 = 2*st.session_state.a_freq*(1-st.session_state.a_freq)*init_larvae
    J_aa_0 = (1-st.session_state.a_freq)**2*init_larvae ##Hardy-Weinberg
    # Generate plots based on the input parameters
    # ...
    X = np.zeros(st.session_state.num_generations)
    Y = np.zeros(st.session_state.num_generations)
    Z = np.zeros(st.session_state.num_generations)
    X[0] = J_AA_0
    Y[0] = J_Aa_0
    Z[0] = J_aa_0
    M_A, M_a, F_A, F_a = attrib_constants(v,st.session_state.rs,rr,st.session_state.sv,st.session_state.sav)
    #####################################################################################
    n_simus = 100
    fig, ax = plt.subplots()
    for j in range(n_simus):
        for n in range(st.session_state.num_generations - 1):
            if X[n] + Y[n] + Z[n] == 0:
                X[n + 1] = 0
                Y[n + 1] = 0
                Z[n + 1] = 0
            else:
                Pf = F_A[n] * (X[n] + Y[n] / 2) / (F_A[n] * (X[n] + Y[n]) + F_a[n] * Z[n])
                Qf = 1 - Pf
                Pm = M_A[n] * (X[n] + Y[n] / 2) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n])
                Qm = 1 - Pm
                p = np.array([[Pf * Pm, Pf * Qm + Pm * Qf, 1 - Pf - Pm]])
                p = p.flatten()
                #print(p.shape)
                offspring = (
                g * (F_A[n] * (X[n] + Y[n]) + F_a[n] * Z[n])
                * np.random.multinomial(st.session_state.N, p)
                / (1 + (X[n] + Y[n] + Z[n]) / M)
                )
                X[n + 1] = offspring[0]
                Y[n + 1] = offspring[1]
                Z[n + 1] = offspring[2]

        tot = X + Y + Z
        fa = (Z + Y / 2) / tot
        ax.plot(np.arange(1, st.session_state.num_generations+1), fa, 'r-', linewidth=0.2)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Frequency of allele a")
    ax.set_xlim([1, st.session_state.num_generations])
    ax.set_ylim([0, 1.001])
    ax.tick_params(axis='both', which='major', labelsize=10)
    st.pyplot(fig)

# Main tab
with st.sidebar:
    main_tab = st.radio("Navigation", ["Main", "Progression of virulence", "Genetic drift", "User guide", "Parameter settings"])

if main_tab == "Main":
    st.markdown("## Main")
    st.session_state.area = st.number_input("Field st.session_state.area (ha):", value=st.session_state.area, step=0.1)
    st.session_state.a_freq = st.slider("Frequency of the virulence allele (%):", min_value=0.0, max_value=100.0, value=st.session_state.a_freq*100, step=0.1)/100
    st.session_state.init_infest = st.slider("Initial infestation (cysts/g of soil):", min_value=0.1, max_value=80.0, value=st.session_state.init_infest, step=0.1)
    st.session_state.deployment_type = st.text_input("Deployment type:", value=st.session_state.deployment_type)
    st.session_state.deployment_type = st.session_state.deployment_type.upper()  # Convert input to uppercase
    if not all(ch in ['R', 'S'] for ch in st.session_state.deployment_type):
        st.error("Invalid deployment type. Please enter a string containing only 'R' or 'S'.")
    update_button = st.button("Update")
    st.session_state.rep_deployment = st.checkbox("Indefinitely repeat this deployment", value=True)
    if st.session_state.rep_deployment:
        st.session_state.num_generations = st.slider("Number of generations (if checked above):", min_value=1, max_value=100, value=st.session_state.num_generations, step=1)
    else:
        st.session_state.num_generations = len(st.session_state.deployment_type)
    st.session_state.bc = st.slider("Efficacy of biocontrol (%):", min_value=0.0, max_value=100.0, value=st.session_state.bc*100, step=0.1)/100
    st.session_state.area = st.session_state.area
    if not st.session_state.rep_deployment:
        st.session_state.num_generations = len(st.session_state.deployment_type)
    # Other parameters
    step = 0.01
    h = 0.88                           #Egg hatching success rate
    rr = 1                             #Avirulent pest sex ratio when resistance
    conversion_factor = 3.3*10**9
    M = 145*st.session_state.N*conversion_factor*st.session_state.area   #Limiting factor
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
    init_larvae = st.session_state.init_infest*st.session_state.N*0.05*h*conversion_factor*st.session_state.area #Cysts x nbre of eggs per cyst * dessication * hatching success * converstion to 1ha field * field st.session_state.area
    J_AA_0 = (1-st.session_state.a_freq)**2*init_larvae
    J_Aa_0 = 2*st.session_state.a_freq*(1-st.session_state.a_freq)*init_larvae
    J_aa_0 = (1-st.session_state.a_freq)**2*init_larvae ##Hardy-Weinberg
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
            X[n+1] = (g * (M_A[n] * F_A[n] * (X[n] + Y[n]/2)**2) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n])) / (1 + (X[n] + Y[n] + Z[n]) / M)
            Y[n+1] = (g * (M_A[n] * (F_a[n] * Z[n] + F_A[n] * Y[n]/2) * (X[n] + Y[n]/2) + F_A[n] * (M_a[n] * Z[n] + M_A[n] * Y[n]/2) * (X[n] + Y[n]/2))) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n]) / (1 + (X[n] + Y[n] + Z[n]) / M)
            Z[n+1] = (g * (F_a[n] * Z[n] + F_A[n] * Y[n]/2) * (M_a[n] * Z[n] + M_A[n] * Y[n]/2)) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n]) / (1 + (X[n] + Y[n] + Z[n]) / M)
    
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
    st.session_state.YY = Y
    st.session_state.ZZ = Z
    st.session_state.ff_A = f_A
    st.session_state.ff_a = f_a
    generate_main_plot(tot)
    if update_button:
        # Generate plots based on updated parameters
        if len(st.session_state.deployment_type) > 1:
            st.session_state.rep_deployment = False
            st.session_state.num_generations = len(st.session_state.deployment_type)
        print(st.session_state.num_generations)
        print(v)
        generate_main_plot(tot)
    st.markdown(f"Basic reproduction number R0 = {R0}")
    st.markdown(f"Suppression threshold = {disp}")
elif main_tab == "Progression of virulence":
    st.markdown("## Progression of virulence")
    # Define your plot for this tab
    generate_virulence_plot(st.session_state.ff_A,st.session_state.ff_a,st.session_state.YY,st.session_state.ZZ)
elif main_tab == "Genetic drift":
    st.markdown("## Genetic drift")
    # Define your plot for this tab
    generate_genetic_drift_plot()
elif main_tab == "User guide":
    st.markdown("## User guide")
    # Edit and add your user guide text here

elif main_tab == "Parameter settings":
    st.markdown("## Parameter settings")
    st.markdown("These parameters describe the basic biology of PCNs. They are retrieved from intensive literature review and cautious estimations. Please edit these settings if and only if you have enough knowledge!!")
    st.session_state.sav = st.slider("Survival of avirulent PCNs on resistant plants (%):", min_value=0.0, max_value=100.0, value=st.session_state.sav*100, step=0.1)/100
    st.session_state.sv = st.slider("Survival of virulent PCNs on resistant plants (%):", min_value=0.0, max_value=100.0, value=st.session_state.sv*100, step=0.1)/100
    st.session_state.rs = st.slider("Average male allocation on susceptible potato (%):", min_value=0.0, max_value=100.0, value=st.session_state.rs*100, step=0.1)/100
    st.session_state.s_E = st.slider("Survival of larvae from cysts (%):", min_value=0.010, max_value=10.0, value=st.session_state.s_E*100, step=0.001)/100
    st.markdown("(Includes suicidal hatching and larval desiccation)")
    st.session_state.N = st.slider("Average eggs per cyst:", min_value=200, max_value=500, value=st.session_state.N, step=1)


# In[ ]:




