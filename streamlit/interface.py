import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_extras.switch_page_button import switch_page

#Functions

def preprocess_plan(file_plan):
    """
    Preprocess plan of the room
        Parameters
        ----------
        file_plan : file
            Uploaded file with room plan
    """
    #TODO
    pass

def preprocess_style(file_style):
    """
    Preprocess style of the room
        Parameters
        ----------
        file_style : file
            Uploaded file with room style
    """
    #TODO
    pass

def preprocess_cost(num_cost):
    if num_cost:
        return num_cost
    else:
        return -1
    
def preprocess_type(room_type):
    if room_type=="Other":
        return -1
    else:
        return room_type




#Layout

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


st.title('Create a room of your dream')

#<p style='font-size:20;'>
room_plan = st.file_uploader("**Upload your room plan**")

if room_plan is not None:
    preprocess_plan(room_plan)

room_style = st.file_uploader("**Upload your room style**")

if room_style is not None:
    preprocess_plan(room_style)
    
cost = st.number_input(
        "**Enter maximum cost of the room**",
        min_value=0,
        value=0,
        help='CHF',
        label_visibility='visible',
    )

room_type=st.radio(
        "**Choose your room type**",
        ["Bedroom", "Livingroom", "Diningroom", "Library","Other"],
        label_visibility='visible',
        disabled=False,
        horizontal=True,
    )

col1, col2, col3 = st.columns(3)

# with col1:
#     st.write(' ')

with col2:
    generate=st.button('Generate',type='primary',use_container_width=True)
    if generate:
        st.markdown('<p style="text-align: center;"><font size=4> Generating... </font></p>', unsafe_allow_html=True)
        switch_page("results")
# with col3:
#     st.write(' ')




# col1, col2 = st.columns(2)
# with col2:
# st.write (room_type)