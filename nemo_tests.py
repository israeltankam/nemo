import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Create the figure and axes
fig, ax = plt.subplots()

# Generate some random data for plotting
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot the data on the main plot
ax.plot(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Main Plot')

# Create two columns with widths in the ratio 2:1
col1, col2 = st.columns([2, 1])

# Full column for the main plot
with col1:
    # Display the main plot using Streamlit's pyplot function
    st.pyplot(fig)

# Second column with upper and lower plots
with col2:
    # Upper plot
    with st.expander("Upper Plot"):
        # Create a new figure and axes
        fig_upper, ax_upper = plt.subplots()

        # Plot the upper plot data
        ax_upper.plot(x, y**2)
        ax_upper.set_xlabel('X')
        ax_upper.set_ylabel('Y^2')
        ax_upper.set_title('Upper Plot')

        # Display the upper plot using Streamlit's pyplot function
        st.pyplot(fig_upper)

    # Lower plot
    with st.expander("Lower Plot"):
        # Create a new figure and axes
        fig_lower, ax_lower = plt.subplots()

        # Plot the lower plot data
        ax_lower.plot(x, np.cos(x))
        ax_lower.set_xlabel('X')
        ax_lower.set_ylabel('Cos(X)')
        ax_lower.set_title('Lower Plot')

        # Display the lower plot using Streamlit's pyplot function
        st.pyplot(fig_lower)
