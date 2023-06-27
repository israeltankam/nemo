#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Set page layout to centered and responsive
# st.set_page_config(layout="centered")
# Set default values of parameters
area = 1.0
a_freq = 5.0
init_infest = 10.0
deployment_type = 'R'
rep_deployment = True
num_generations = 20
bc = 0.0
sav = 25.0/100
sv = 25.0/100
rs = 35.0/100
s_E = 0.036
N = 350
step = 0.01
h = 0.88                       
rr = 1                              
conversion_factor = 3.3*10**9
M = 145*N*conversion_factor*area
s = 1
v = [1] * num_generations

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

def generate_deployment_vector(input_string, n_gen):   #n_gen will be the global num_generations
    if input_string == 'R':
        return [1] * n_gen
    elif input_string == 'S':
        return [0] * n_gen
    else:
        vector = [1 if char == 'R' else 0 for char in input_string]
        return vector
      
def generate_main_plot(tot):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, num_generations+1), tot, '-r')
    ax.set_xlabel("Generations")
    ax.set_ylabel("PCNs in field (log scale)")
    ax.set_xlim([1, num_generations])
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=10)
    # Display the plot in Streamlit
    st.pyplot(fig)


def generate_virulence_plot(f_A,f_a,Y,Z):
    st.pyplot()

def generate_genetic_drift_plot():
    st.pyplot()

# Main tab
with st.sidebar:
    main_tab = st.radio("Navigation", ["Main", "Progression of virulence", "Genetic drift", "User guide", "Parameter settings"])

if main_tab == "Main":
    st.markdown("## Main")
    area = st.number_input("Field area (ha):", value=1.0, step=0.1)
    a_freq = st.slider("Frequency of the virulence allele (%):", min_value=0.0, max_value=100.0, value=5.0, step=0.1)/100
    init_infest = st.slider("Initial infestation (cysts/g of soil):", min_value=0.1, max_value=80.0, value=10.0, step=0.1)
    deployment_type = st.text_input("Deployment type:", value='R')
    deployment_type = deployment_type.upper()  # Convert input to uppercase
    if not all(ch in ['R', 'S'] for ch in deployment_type):
        st.error("Invalid deployment type. Please enter a string containing only 'R' or 'S'.")
    update_button = st.button("Update")
    rep_deployment = st.checkbox("Indefinitely repeat this deployment", value=True)
    if rep_deployment:
        num_generations = st.slider("Number of generations (if checked above):", min_value=1, max_value=100, value=20, step=1)
    else:
        num_generations = len(deployment_type)
    bc = st.slider("Efficacy of biocontrol (%):", min_value=0.0, max_value=100.0, value=0.0, step=0.1)/100
    area = area
    if not rep_deployment:
        num_generations = len(deployment_type)
    # Other parameters
    step = 0.01
    h = 0.88                           #Egg hatching success rate
    rr = 1                             #Avirulent pest sex ratio when resistance
    conversion_factor = 3.3*10**9
    M = 145*N*conversion_factor*area   #Limiting factor
    v = generate_deployment_vector(deployment_type, num_generations)
    s = 1
    g = h*N*s_E*s*(1-bc)
    alpha = sv*rs/sav
    R0 = g*sv*(1-rs)
    if np.all(v == np.ones(num_generations)):
        thres = dec2(2 * (1 - alpha))
    elif np.all(v == np.zeros(num_generations)):
        thres = 1
    else:
        thres = float('-inf')
    init_larvae = init_infest*N*0.05*h*conversion_factor*area #Cysts x nbre of eggs per cyst * dessication * hatching success * converstion to 1ha field * field area
    J_AA_0 = (1-a_freq)**2*init_larvae
    J_Aa_0 = 2*a_freq*(1-a_freq)*init_larvae
    J_aa_0 = (1-a_freq)**2*init_larvae ##Hardy-Weinberg
    # Generate plots based on the input parameters
    # ...
    X = np.zeros(num_generations)
    Y = np.zeros(num_generations)
    Z = np.zeros(num_generations)
    X[0] = J_AA_0
    Y[0] = J_Aa_0
    Z[0] = J_aa_0
    M_A, M_a, F_A, F_a = attrib_constants(v,rs,rr,sv,sav)

    for n in range(num_generations-1):
        if X[n] + Y[n] + Z[n] == 0:
            X[n+1] = 0
            Y[n+1] = 0
            Z[n+1] = 0
        else:
            X[n+1] = (g * (M_A[n] * F_A[n] * (X[n] + Y[n]/2)**2) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n])) / (1 + (X[n] + Y[n] + Z[n]) / M)
            Y[n+1] = (g * (M_A[n] * (F_a[n] * Z[n] + F_A[n] * Y[n]/2) * (X[n] + Y[n]/2) + F_A[n] * (M_a[n] * Z[n] + M_A[n] * Y[n]/2) * (X[n] + Y[n]/2))) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n]) / (1 + (X[n] + Y[n] + Z[n]) / M)
            Z[n+1] = (g * (F_a[n] * Z[n] + F_A[n] * Y[n]/2) * (M_a[n] * Z[n] + M_A[n] * Y[n]/2)) / (M_A[n] * (X[n] + Y[n]) + M_a[n] * Z[n]) / (1 + (X[n] + Y[n] + Z[n]) / M)
    
    tot = X + Y + Z
    f_AA = np.zeros(num_generations)
    f_Aa = np.zeros(num_generations)
    f_aa = np.zeros(num_generations)
    f_A = np.zeros(num_generations)
    f_a = np.zeros(num_generations)

    for n in range(num_generations):
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
    generate_main_plot(tot)
    if update_button:
        # Generate plots based on updated parameters
        if len(deployment_type) > 1:
            rep_deployment = False
            num_generations = len(deployment_type)
        print(num_generations)
        print(v)
        generate_main_plot(tot)
    st.markdown(f"Basic reproduction number R0 = {R0}")
    st.markdown(f"Suppression threshold = {disp}")
elif main_tab == "Progression of virulence":
    st.markdown("## Progression of virulence")
    # Define your plot for this tab

elif main_tab == "Genetic drift":
    st.markdown("## Genetic drift")
    # Define your plot for this tab

elif main_tab == "User guide":
    st.markdown("## User guide")
    # Edit and add your user guide text here

elif main_tab == "Parameter settings":
    st.markdown("## Parameter settings")
    st.markdown("These parameters describe the basic biology of PCNs. They are retrieved from intensive literature review and cautious estimations. Please edit these settings if and only if you have enough knowledge!!")
    sav = st.slider("Survival of avirulent PCNs on resistant plants (%):", min_value=0.0, max_value=100.0, value=25.0, step=0.1)/100
    sv = st.slider("Survival of virulent PCNs on resistant plants (%):", min_value=0.0, max_value=100.0, value=25.0, step=0.1)/100
    rs = st.slider("Average male allocation on susceptible potato (%):", min_value=0.0, max_value=100.0, value=35.0, step=0.1)/100
    s_E = st.slider("Survival of larvae from cysts:", min_value=0.010, max_value=0.100, value=0.036, step=0.001)
    st.markdown("(Includes suicidal hatching and larval desiccation)")
    N = st.slider("Average eggs per cyst:", min_value=200, max_value=500, value=350, step=1)


# In[ ]:




