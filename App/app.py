import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# Load the data


def load_data():
    data = pd.read_csv("data/dataset.csv")
    return data


data = load_data()


def get_radar_chart(input_data):
    categories = ['processing cost', 'mechanical properties', 'chemical stability',
                  'thermal stability', 'device integration']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[1, 5, 2, 2, 3],
        theta=categories,
        fill='toself',
        name='Product A'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[4, 3, 2.5, 1, 2],
        theta=categories,
        fill='toself',
        name='Product B'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=False
    )

    return fig


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
        st.write('PLease connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app makes predictions using a Machine Learning Model whether a breast mass is benign or malignant based in the measurements it receives from your lab. You can also update the measurements manually using the sliders in the sidebar.')

    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        st.write('This is column2')


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
