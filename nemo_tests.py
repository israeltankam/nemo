import streamlit as st
from streamlit import session_state
import hydralit_components as hc

#make it look nice from the start
st.set_page_config(layout='wide',initial_sidebar_state='collapsed')
st.session_state.setdefault("deployment_type", "")
# specify the primary menu definition
menu_data = [
    {'icon': "far fa-copy", 'label':"Model & Parameters"},
    {'icon': "far fa-chart-bar", 'label':"Simulations"},#no tooltip message
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
if main_tab == "Model & Parameters":

    st.markdown("Click to select a type of crop to deploy. Do not begin nor end with Non-host.")

    col1, col2, col3, col4, col5 = st.columns(5)
    
    if col1.button("Susceptible"):
        st.session_state.deployment_type += "S"

    if col2.button("Resistant"):
        st.session_state.deployment_type += "R"

    if col3.button("Non-host"):
        st.session_state.deployment_type += "N"

    if col4.button("Delete"):
        if len(st.session_state.deployment_type) > 0:
            st.session_state.deployment_type = st.session_state.deployment_type[:-1]

    if col5.button("Erase"):
        st.session_state.deployment_type = ""

    # Display the deployment type
    st.markdown("Deployment type")
    
    st.text(st.session_state.deployment_type)

#get the id of the menu item clicked
#st.info(f"{menu_id}")