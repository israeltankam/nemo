import streamlit as st
import hydralit_components as hc

#make it look nice from the start
st.set_page_config(layout='wide',initial_sidebar_state='collapsed')

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
    st.markdown("Here am I")
#get the id of the menu item clicked
#st.info(f"{menu_id}")