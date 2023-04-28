import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from joblib import load

scaler = load('scaler.joblib')
model = load('model.joblib')


# Load the data


def load_data():
    data = pd.read_csv("data/dataset.csv")
    return data


data = load_data()

# To make sure all predictions values are scaled between 0 and 1


def get_scaled_values(input_dict):

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter',
                  'Area', 'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
           input_data['fractal_dimension_mean']
           ],
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
           input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
           input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Worst'
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def add_predictions(input_data):
    # Load scaler
    scaler = load('scaler.joblib')

    # Load model
    model = load('model.joblib')

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array).reshape(1, -1)

    prediction = model.predict(input_scaled)

    st.subheader('Cell Cluster Prediction')

    if prediction[0] == 0:
        st.success('Great news: This mass is **benign**.')

        # Get probability estimate for benign
        proba = model.predict_proba(input_scaled)[0][0]
        st.write('The probability of this mass being **benign** is:', proba)

    else:
        st.error(
            'Warning: This mass is **malignant**. Please consult a doctor immediately!')

        # Get probability estimate for malignant
        proba = model.predict_proba(input_scaled)[0][1]
        st.write('The probability of this mass being **malignant** is:', proba)


def app():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor",
        layout='wide',
        initial_sidebar_state='expanded'
    )

    input_data = add_sidebar()

    # st.write(input_data)

    # Creating a slider for each predictor about 30 of them, since dropped 2
    with st.container():
        st.title('Breast Cancer Predictor')
        st.write('Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app makes predictions using a Machine Learning Model to determine if a breast mass is benign or malignant based in the measurements it receives from your lab. You can also update the measurements manually using the sliders in the sidebar.')

    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)


def add_sidebar():
    st.sidebar.header('Cell Nuclei Measurements')

    data = load_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    # Loop through each labels
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())

        )
    return input_dict


if __name__ == '__main__':
    app()
