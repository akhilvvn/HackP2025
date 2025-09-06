import streamlit as st
from PIL import Image
import pandas as pd
from io import BytesIO
from object_search import search_query_image, detect_query_objects

st.set_page_config(page_title="Object-Level Search", layout="wide")
st.title("Object-Level Image Search")

uploaded_file = st.file_uploader("Upload a query image", type=["png", "jpg", "jpeg"])

threshold = st.slider("Similarity Threshold", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Query Image", width=300)

    detected_objects = detect_query_objects(uploaded_file)

    if not detected_objects:
        st.warning("No objects detected in the query image.")
    else:
        selected_class = st.selectbox(
            "Select an object class to search for:",
            options=list(detected_objects.keys())
        )

        if st.button("Search"):
            st.write(f"Searching for images containing: **{selected_class}** with threshold **{threshold}** ...")
            top_matches = search_query_image(uploaded_file, selected_class=selected_class, threshold=threshold)

            results_data = []

            if top_matches:
                images_per_row = 4
                for i in range(0, len(top_matches), images_per_row):
                    row_matches = top_matches[i:i+images_per_row]
                    cols = st.columns(len(row_matches))
                    for col, (img_path, score) in zip(cols, row_matches):
                        col.image(Image.open(img_path), caption=f"Score: {score:.2f}", width=200)
                        with col.expander("Enlarge"):
                            st.image(Image.open(img_path), caption=f"Full-size image: {score:.2f}")

                        results_data.append({
                            "query_image": uploaded_file.name,
                            "selected_class": selected_class,
                            "match_rank": len(results_data)+1,
                            "matched_image": img_path.split("/")[-1],
                            "similarity_score": float(score)
                        })

                df_results = pd.DataFrame(results_data)
                csv_buffer = BytesIO()
                df_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="object_search_results.csv",
                    mime="text/csv"
                )
            else:
                st.write(f"No matching images found for class **{selected_class}** above threshold.")
