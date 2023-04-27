import streamlit as st
import pandas as pd


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor",
        layout='wide',
        initial_sidebar_state='expanded'
    )

    with st.container():
        st.title('Breast Cancer Predictor')
        st.write('PLease connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app makes predictions using a Machine Learning Model whether a breast mass is benign or malignant based in the measurements it receives from your lab. You can also update the measurements manually using the sliders in the sidebar.')


if __name__ == '__main__':
    main()
