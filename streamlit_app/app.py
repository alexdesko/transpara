import streamlit as st
from dotenv import load_dotenv; load_dotenv()

st.set_page_config(page_title="Transpara Demo", layout="wide")


def main():
    st.title("Transpara: Chest X‑ray Classification Demo")
    st.markdown(
        """
        Explore trained runs, try the model on your own image, and visualize
        Grad‑CAM explanations. Use the pages in the left sidebar to navigate.

        - Try Model: upload an image, view predictions
        - Explainability: Grad‑CAM heatmap (ResNet18)
        - Training Runs: browse metrics and config for past runs
        """
    )


if __name__ == "__main__":
    main()
