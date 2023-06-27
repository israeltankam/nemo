import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Set Streamlit app title
st.title("Nemo")

# Set page layout to centered and responsive
#st.set_page_config(layout="centered")

# Define default parameter values
st.session_state.setdefault("area", 1.0)
st.session_state.setdefault("a_freq", 5.0)
st.session_state.setdefault("init_infest", 10.0)
st.session_state.setdefault("deployment_type", "R")
st.session_state.setdefault("rep_deployment", True)
st.session_state.setdefault("num_generations", 20)
st.session_state.setdefault("bc", 0.0)
st.session_state.setdefault("sav", 25.0)
st.session_state.setdefault("sv", 25.0)
st.session_state.setdefault("rs", 35.0)
st.session_state.setdefault("s_E", 0.036)
st.session_state.setdefault("N", 350)

# Define dummy plots
def plot_dummy_1():
    x = np.linspace(0, 10, 100)
    y = (st.session_state.a_freq ** 2) * x
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dummy Plot 1')
    st.pyplot()

def plot_dummy_2():
    x = np.linspace(0, 10, 100)
    y = (st.session_state.sav * st.session_state.rs) * x + st.session_state.s_E
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dummy Plot 2')
    st.pyplot()

def plot_dummy_3():
    x = np.linspace(0, 10, 100)
    y = st.session_state.init_infest ** 2 + x - st.session_state.bc
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dummy Plot 3')
    st.pyplot()

with st.sidebar:
    main_tab = st.radio("Navigation", ["Main", "Progression of virulence", "Genetic drift", "User guide", "Parameter settings"])
    
# Main tab
if main_tab == "Main":
    st.header("Main")
    st.session_state.area = st.number_input("Field area (ha):", value=st.session_state.area, step=0.1)
    st.session_state.a_freq = st.slider("Frequency of the virulence allele (%):", min_value=0.0, max_value=100.0, value=st.session_state.a_freq, step=0.1)
    st.session_state.init_infest = st.slider("Initial infestation (cysts/g of soil):", min_value=0.1, max_value=80.0, value=st.session_state.init_infest, step=0.1)
    st.session_state.deployment_type = st.text_input("Deployment type:", value=st.session_state.deployment_type).upper()
    st.session_state.rep_deployment = st.checkbox("Indefinitely repeat this deployment", value=st.session_state.rep_deployment)
    if st.session_state.rep_deployment:
        st.session_state.num_generations = st.slider("Number of generations (if checked above):", min_value=1, max_value=100, value=st.session_state.num_generations, step=1)
    st.session_state.bc = st.slider("Efficacy of biocontrol (%):", min_value=0.0, max_value=100.0, value=st.session_state.bc, step=0.1)

    # Display updated global parameters
    st.sidebar.write("Global Parameters:")
    st.sidebar.write(f"Field area (ha): {st.session_state.area}")
    st.sidebar.write(f"Frequency of the virulence allele (%): {st.session_state.a_freq}")
    st.sidebar.write(f"Initial infestation (cysts/g of soil): {st.session_state.init_infest}")
    st.sidebar.write(f"Deployment type: {st.session_state.deployment_type}")
    st.sidebar.write(f"Indefinitely repeat this deployment: {st.session_state.rep_deployment}")
    if st.session_state.rep_deployment:
        st.sidebar.write(f"Number of generations (if checked above): {st.session_state.num_generations}")
    st.sidebar.write(f"Efficacy of biocontrol (%): {st.session_state.bc}")

    # Display plot
    plot_dummy_1()

# Progression of virulence tab
if main_tab == "Progression of virulence":
    st.header("Progression of virulence")

    # Display plot
    plot_dummy_2()

# Genetic drift tab
if main_tab == "Genetic drift":
    st.header("Genetic drift")

    # Display plot
    plot_dummy_3()

# User guide tab
if main_tab == "User guide":
    st.header("User guide")
    # Edit and add your user guide text here

# Parameter settings tab
if main_tab == "Parameter settings":
    st.header("Parameter settings")
    st.write("These parameters describe the basic biology of PCNs. They are retrieved from intensive literature review and cautious estimations. Please edit these settings only if you have enough knowledge!")

    st.slider("Survival of avirulent PCNs on resistant plants (%):", min_value=0.0, max_value=100.0, value=st.session_state.sav, step=0.1)
    st.slider("Survival of virulent PCNs on resistant plants (%):", min_value=0.0, max_value=100.0, value=st.session_state.sv, step=0.1)
    st.slider("Average male allocation on susceptible potato (%):", min_value=0.0, max_value=100.0, value=st.session_state.rs, step=0.1)
    st.slider("Survival of larvae from cysts:", min_value=0.010, max_value=0.100, value=st.session_state.s_E, step=0.001)
    st.slider("Average eggs per cyst:", min_value=200, max_value=500, value=st.session_state.N, step=1)
