import streamlit as st
import plotly.express as px
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import base64
from PIL import Image
import cv2
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

def load_xlsx(selected_lokasi):
    file_path = f"C:/Users/David/Downloads/amin/Dataset_Backup/lokasi_kereta_{selected_lokasi}.xlsx"  # Assuming CSV files are named as lokasi_kereta_1.csv, lokasi_2.csv, etc.
    df = pd.read_excel(file_path)
    # Do something with the loaded CSV file
    st.write("Loaded CSV file:", file_path)
    st.write(df.head())  # Just displaying the first few rows of the dataframe

# Get the current working directory
cwd = os.getcwd()

# Specify the relative path to your file from the cwd
file_path = os.path.join(cwd, "C:/Users/David/Downloads/amin/Dataset Backup")

st.set_page_config(page_title="Dashboard", page_icon=":bar_chart:", layout="wide")


#File Upload
with st.container():
    st.subheader("Welcome,")
    st.title("5G Beam Prediction with Machine Learning on High Speed Train with Machine Learning Dashboard")
    
# Buat tempat upload File
fl = st.file_uploader(":file_folder: Upload a result file", type=(["csv","txt","xlsx","xls"]))

#selected_columns = st.multiselect('Show result',['Outage Probability', 'Power Footprint', 'Algorithmic Comparison'])
#total_columns = len(selected_columns)

# Column
with st.container():
    st.write("---")
    # left_column,middle_column, right_column = st.columns(total_columns) # Generate 2 column
    left_column, right_column = st.columns(2) #Generate 2 column
    with left_column:
        st.header("Lokasi Kereta")
        selected_lokasi = st.selectbox("Pilih Kereta", options=list(range(1,39)), index=0)
        file_lokasi = ["lokasi_kereta_" + str(i) for i in range(1, 39)]
        file_path = f"C:/Users/David/Downloads/amin/Dataset/lokasi_kereta_{selected_lokasi}.xlsx" 
        
        # Load data
        df = pd.read_excel(file_path)

        # Circle parameters
        circle_x = 0
        circle_y = 0
        circle_radius = 350

        custom_marker_path = "C:/Users/David/Downloads/amin/565410.png"  # Replace "path_to_your_image.png" with the path to your image
        custom_marker = Image.open(custom_marker_path)
        custom_image = Image.open('C:/Users/David/Downloads/amin/565410.png')
        fig_kereta = px.scatter(df,x='y', y='x', animation_frame='time')


        # Add circle
        fig_kereta.add_shape(
            type='circle',
            xref='x',
            yref='y',
            x0=circle_x - circle_radius,
            y0=circle_y - circle_radius,
            x1=circle_x + circle_radius,
            y1=circle_y + circle_radius,
            line=dict(color='red'))
        
        fig_kereta.add_layout_image(
            dict(
                source=Image.open(f'C:/Users/David/Downloads/amin/565410.png'),
                x='x',  # x-coordinate of the image
                y='y',  # y-coordinate of the image
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                sizex=1,  # width of the image
                sizey=1,  # height of the image
                sizing="contain",
                opacity=1,
                layer="above"))
    
        fig_kereta.update_layout(
            width=500,  # Set width in pixels
            height=500,
            font=dict(
            size=30),
            xaxis=dict(
                title="X Axis",
                titlefont_size=20,
                tickfont_size=20),
            yaxis=dict(
                title="Y Axis",
                titlefont_size=20,
                tickfont_size=20))
        st.plotly_chart(fig_kereta)
        
        
       
        # Add scatter plot with custom image marker
        # Load custom image
        #custom_marker_path = "C:/Users/David/Downloads/amin/565410.png"  # Replace "path_to_your_image.png" with the path to your image
        #custom_marker = Image.open(custom_marker_path)
        #fig_kereta.add_trace(go.Scatter(
        #    marker=dict(
        #    x='x',
        #    y='y',
        #    mode='markers',
        #    marker=dict(
        #        symbol='star',  # Use custom image as marker
        #        color='blue',
        #        size=10,
        #        opacity=0.5,
        #       line=dict(
        #            width=2,
        #            color='DarkSlateGrey'),
        #        image=custom_marker,
        #        sizemode='diameter'
        #    )
        #))



    with right_column:
        #st.subheader("Index")
        #input_index = "C:/Users/David/Downloads/amin/combined_index.xlsx"
        #df_index = pd.read_excel(input_index)
        #fig = px.line(df, x='Index_1', y=None,
        #      title='Combined Data Graph')
        st.header("Beam Index")
        selected_index = st.selectbox("Pilih Beam Index", options=list(range(1,39)), index=0)
        #file_index = ["indeks" + str(i) for i in range(1, 39)]
        #file_path = f"C:/Users/David/Downloads/amin/Dataset/indeks{selected_index}.xlsx" 


        # Read train coordinates and time data from Excel file
        file_path_train = "C:/Users/David/Downloads/amin/Dataset/indeks1.xlsx"
        df_train = pd.read_excel(file_path_train)

        # Define base station coordinates
        base_station_x = 0
        base_station_y = -300

        # Define beam length
        beam_length = 100  # Example length

        # Create Plotly figure
        fig = px.scatter(df_train, x='y', y='x', title='Train with Beam Pattern')

        # Add base station
        fig.add_trace(go.Scatter(x=[base_station_x], y=[base_station_y], mode='markers', marker=dict(color='red', size=10)))

        # Add initial beam pattern
        closest_train_x = df_train.loc[0, 'y']
        closest_train_y = df_train.loc[0, 'x']
        angle_to_closest_train = np.arctan2(closest_train_y - base_station_y, closest_train_x - base_station_x)
        angle1 = angle_to_closest_train + np.pi / 6
        angle2 = angle_to_closest_train - np.pi / 6
        tip1_x = closest_train_x + beam_length * np.cos(angle1)
        tip1_y = closest_train_y + beam_length * np.sin(angle1)
        tip2_x = closest_train_x + beam_length * np.cos(angle2)
        tip2_y = closest_train_y + beam_length * np.sin(angle2)
        beam_shape_x = [base_station_x, tip1_x, tip2_x, base_station_x]
        beam_shape_y = [base_station_y, tip1_y, tip2_y, base_station_y]
        fig.add_trace(go.Scatter(x=beam_shape_x, y=beam_shape_y, mode='lines', fill='toself', fillcolor='rgba(0, 0, 255, 0.2)', line=dict(color='blue'), name='Beam Pattern'))

        # Update layout
        fig.update_layout(
            xaxis_title='X',
            yaxis_title='Y',
            xaxis=dict(range=[-400, 400]),  # Set fixed range for x-axis
            yaxis=dict(range=[-400, 400]),  # Set fixed range for y-axis
            showlegend=True,
            width=1000,
            height=500,
            updatemenus=[{
                "buttons": [{"args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                            "label": "Play",
                            "method": "animate"}],
                "direction": "left",
                "pad": {"r":250, "t": 80},
                "showactive": False,
                "type": "buttons",
                "x": 1,
                "xanchor": "right",
                "y": 0.1,
                "yanchor": "top"
            }]
        )

        # Define frames for animation
        frames = []
        for i, (_, row) in enumerate(df_train.iterrows()):
            closest_train_x = row['y']
            closest_train_y = row['x']
            angle_to_closest_train = np.arctan2(closest_train_y - base_station_y, closest_train_x - base_station_x)
            angle1 = angle_to_closest_train + np.pi / 6
            angle2 = angle_to_closest_train - np.pi / 6
            tip1_x = closest_train_x + beam_length * np.cos(angle1)
            tip1_y = closest_train_y + beam_length * np.sin(angle1)
            tip2_x = closest_train_x + beam_length * np.cos(angle2)
            tip2_y = closest_train_y + beam_length * np.sin(angle2)
            beam_shape_x = [base_station_x, tip1_x, tip2_x, base_station_x]
            beam_shape_y = [base_station_y, tip1_y, tip2_y, base_station_y]
            frames.append(go.Frame(data=[
                go.Scatter(x=[base_station_x], y=[base_station_y], mode='markers', marker=dict(color='red', size=10)),
                go.Scatter(x=[closest_train_x], y=[closest_train_y], mode='markers', marker=dict(color='blue')),
                go.Scatter(x=beam_shape_x, y=beam_shape_y, mode='lines', fill='toself', fillcolor='rgba(0, 0, 255, 0.2)', line=dict(color='blue'), name='Beam Pattern')],
                name=i))

        # Add frames to the figure
        fig.frames = frames

        # Display the plot using Streamlit
        st.plotly_chart(fig)





# Results
with st.container():
    st.write("---")
    left_column1,left_column2, right_column1,right_column2,edge = st.columns(5) #Generate 2 column
    with left_column1:
        st.subheader("KNN")
        # Read data from Excel file
        excel_file = 'C:/Users/David/Downloads/amin/sementara.xlsx'
        data1 = pd.read_excel(excel_file)

        # Extract top 1 to top 5 results of KNN
        nn_results = data1.loc[data1['Method'] == "KNN", "top_1":"top_5"].values.tolist()[0]

        # Display the top 1 to top 5 results with numbers
        st.write("Top 1 to Top 5 Results for KNN:")
        for i, result in enumerate(nn_results, start=1):
            st.write(f"{i}. {result}")


    with left_column2:
        st.subheader("Lookup Table")
        # Extract top 1 to top 5 results of LT
        nn_results = data1.loc[data1['Method'] == "LT", "top_1":"top_5"].values.tolist()[0]

        # Display the top 1 to top 5 results with numbers
        st.write("Top 1 to Top 5 Results for LT:")
        for i, result in enumerate(nn_results, start=1):
            st.write(f"{i}. {result}")



    with right_column1:
        st.subheader("Neural Network")
        # Extract top 1 to top 5 results of NN
        nn_results = data1.loc[data1['Method'] == "NN", "top_1":"top_5"].values.tolist()[0]

        # Display the top 1 to top 5 results with numbers
        st.write("Top 1 to Top 5 Results for NN:")
        for i, result in enumerate(nn_results, start=1):
            st.write(f"{i}. {result}")

    
    with right_column2:
        st.subheader("Adaboost")
        # Extract top 1 to top 5 results of Adaboost
        nn_results = data1.loc[data1['Method'] == "Ada", "top_1":"top_5"].values.tolist()[0]

        # Display the top 1 to top 5 results with numbers
        st.write("Top 1 to Top 5 Results for Adaboost:")
        for i, result in enumerate(nn_results, start=1):
            st.write(f"{i}. {result}")
    
    with edge:
        st.subheader("RF")
        # Extract top 1 to top 5 results of RF
        nn_results = data1.loc[data1['Method'] == "RF", "top_1":"top_5"].values.tolist()[0]

        # Display the top 1 to top 5 results with numbers
        st.write("Top 1 to Top 5 Results for Adaboost:")
        for i, result in enumerate(nn_results, start=1):
            st.write(f"{i}. {result}")



st.write("---")



