import streamlit as st
from PIL import Image
import os
import pyvista as pv
from stpyvista import stpyvista
from glob import glob

# ipythreejs does not support scalar bars :(
pv.global_theme.show_scalar_bar = False

## Initialize a plotter object
pv.start_xvfb()

# Set the folder path where .obj files are located
pattern_obj = "tikhon_bedroom/*.obj"
pattern_tex = "tikhon_bedroom/*.png"

# Create an empty plotter
plotter = pv.Plotter(window_size=[600,600])

# Loop through each .obj file
for file_obj, file_tex in zip(sorted(glob(pattern_obj)), sorted(glob(pattern_tex))):
    
    # Load the .obj file
    mesh = pv.read(file_obj)
    tex = pv.read_texture(file_tex)
    
    # Add the mesh to the plotter
    tex = pv.numpy_to_texture(tex.to_array()[..., :-1]) if tex.to_array().shape[-1] !=3 else tex
    plotter.add_mesh(mesh, texture=tex)

## Final touches
plotter.view_isometric()
plotter.background_color = '#F4EDE5'


st.set_page_config(initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    image = Image.open('logo-hacky-removebg-preview.png')
    st.image(image,use_column_width=True)

st.title('Your dream room')


## Send to streamlit
stpyvista(plotter, key="pv_cube")