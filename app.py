import streamlit as st
import os
import subprocess
import json
import zipfile
import shutil
# import base64

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Model Training & Inspection")
st.title("Simple Image Classification Web App - Test")
st.write("---")

# Initialize session state variables
if 'model_save_path' not in st.session_state:
    st.session_state['model_save_path'] = "cloud_models/trained_classification_model.keras"
if 'training_result' not in st.session_state:
    st.session_state['training_result'] = None

TEMP_CLOUD_ROOT = "cloud_temp_data"


# 1. Training Logic (Adjusted for Simple Classification Data Structure)
@st.cache_resource(show_spinner="Training classification model... please wait! ⏳")
def train_model_function(data_dir, save_path, epochs, img_size, batch_size):
    # NOTE: data_dir will now be the path to the extracted root folder (containing 'good' and 'defective')
    st.info(f"Starting training with data from: **{data_dir}**")

    # Define the environment variables for the subprocess
    env = os.environ.copy()
    env['DATA_DIR'] = data_dir  # Pass the root folder path (e.g., cloud_temp_data/extracted_data)
    env['MODEL_SAVE_PATH'] = save_path
    env['EPOCHS'] = str(epochs)

    try:
        process = subprocess.run(
            ['python', 'train_model.py'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        st.code(process.stdout, language='text')

        if os.path.exists(save_path):
            return f"Classification Model trained and saved at {save_path}"
        else:
            st.error("Training finished, but model file was not found. Check the script output.")
            return None

    except subprocess.CalledError as e:
        st.error("❌ **Training failed!** Check the error log below.")
        st.code(e.stderr, language='text')
        return None
    except FileNotFoundError:
        st.error("❌ **Error:** Could not find 'train_model.py' or 'python' interpreter.")
        return None


# Split App into Tabs
tab_train, tab_inspect = st.tabs(["🚀 Model Training", "🔍 Unit Inspection"])

# =========================================================================
# 🚀 MODEL TRAINING TAB (Cloud Logic Implemented for Classification)
# =========================================================================
with tab_train:
    st.header("1. Configure Training Parameters")

    # --- A. Data Input (File Uploader) ---
    uploaded_zip_file = st.file_uploader(
        "Upload Training Data (.zip file)",
        type=['zip'],
        help="The ZIP file must contain two subdirectories at the root: **'good'** and **'defective'**."
    )

    with st.form("training_config_form"):
        # --- B. Model Save Path (Internal to Cloud) ---
        model_save_path_local = st.text_input(
            "💾 **Model Save Path (Internal)**",
            value=st.session_state['model_save_path'],
            key="model_path_input",
            help="This is the path on the cloud server where the model will be saved during the session."
        )

        # Advanced Configuration (Optional)
        with st.expander("Advanced Configuration"):
            epochs = st.slider("Number of Epochs", 1, 50, 10)
            batch_size = st.slider("Batch Size", 8, 64, 8)
            img_height = st.number_input("Image Height (px)", 32, 512, 128)
            img_width = st.number_input("Image Width (px)", 32, 512, 128)

        st.write("---")
        submitted = st.form_submit_button("🚀 **Start Classification Training**", type="primary")

    if submitted:
        if uploaded_zip_file is None:
            st.error("⚠️ Please upload a ZIP file containing the training dataset.")
            st.session_state['training_result'] = None

        else:
            os.makedirs(TEMP_CLOUD_ROOT, exist_ok=True)

            # --- 1. Save and Extract ZIP Data ---
            temp_zip_path = os.path.join(TEMP_CLOUD_ROOT, "uploaded_data.zip")
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_zip_file.getbuffer())

            try:
                with st.spinner("Extracting data..."):
                    extracted_data_dir = os.path.join(TEMP_CLOUD_ROOT, "extracted_data")
                    if os.path.exists(extracted_data_dir):
                        shutil.rmtree(extracted_data_dir)
                    os.makedirs(extracted_data_dir)

                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extracted_data_dir)

                        # --- VALIDATION CHECK for Classification Folders ---
                good_dir = os.path.join(extracted_data_dir, "good")
                defective_dir = os.path.join(extracted_data_dir, "defective")

                if not os.path.isdir(good_dir) or not os.path.isdir(defective_dir):
                    st.error(
                        "❌ Extraction failed: Could not find **'good'** and **'defective'** folders in the ZIP file's root.")
                    st.session_state['training_result'] = None

                # --- 2. Ensure the model save directory exists ---
                save_dir = os.path.dirname(model_save_path_local)
                os.makedirs(save_dir, exist_ok=True)

                # --- 3. Call the training function ---
                # The data_dir passed is the root folder containing the class subdirectories
                st.session_state['training_result'] = train_model_function(
                    extracted_data_dir,
                    model_save_path_local,
                    epochs,
                    (img_height, img_width),
                    batch_size
                )

            except Exception as e:
                st.error(f"❌ Error during data extraction or training: {e}")
                st.session_state['training_result'] = None

            finally:
                # --- 4. Cleanup temporary data files ---
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
                if os.path.exists(extracted_data_dir):
                    shutil.rmtree(extracted_data_dir)

                    # --- C. Model Output (Download Button) ---
    if st.session_state['training_result'] and os.path.exists(model_save_path_local):
        st.balloons()
        st.markdown(f"### 🎉 Success: {st.session_state['training_result']}")

        with open(model_save_path_local, "rb") as file:
            st.download_button(
                label="⬇️ **Download Trained Model (.keras)**",
                data=file,
                file_name=os.path.basename(model_save_path_local),
                mime='application/octet-stream',
                type="primary",
                help="Download the model to use for future inspection sessions."
            )

# =========================================================================
# 🔍 UNIT INSPECTION TAB (File Uploader Logic Remains)
# =========================================================================
with tab_inspect:
    st.header("1. Provide Input")

    col_model, col_image = st.columns(2)

    # --- A. Model Input (File Uploader) ---
    with col_model:
        uploaded_model_file = st.file_uploader(
            "📂 **Upload Trained Model (.keras)**",
            type=['keras', 'h5'],
            key="inspection_model_uploader",
            help="Upload a model trained previously or downloaded from the training tab."
        )

    # --- B. Image Input (File Uploader) ---
    with col_image:
        uploaded_image = st.file_uploader(
            "🖼️ **Upload Image for Inspection**",
            type=['jpg', 'jpeg', 'png'],
            key="inspection_image_uploader"
        )

    st.write("---")
    st.header("2. Inspection Result")

    if uploaded_image is not None and uploaded_model_file is not None and st.button("🔍 **Run Inspection**"):

        TEMP_INSPECT_DIR = "temp_inspection"
        os.makedirs(TEMP_INSPECT_DIR, exist_ok=True)

        # 1. Save uploaded model temporarily on the cloud server
        temp_model_path = os.path.join(TEMP_INSPECT_DIR, "uploaded_model.keras")
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_model_file.getbuffer())

        # 2. Save uploaded image temporarily on the cloud server
        temp_image_path = os.path.join(TEMP_INSPECT_DIR, uploaded_image.name)
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # --- Run the Inspection Subprocess ---
        with st.spinner("Running inspection..."):
            try:
                process = subprocess.run(
                    ['python', 'inspect_unit.py', temp_image_path, temp_model_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                result_json = process.stdout.strip()
                inspection_result = json.loads(result_json)

                # --- Display Results ---
                if inspection_result["success"]:

                    pred_class = inspection_result["prediction_class"]
                    confidence = inspection_result.get("confidence", "N/A")  # Use .get() for robustness

                    st.success("✅ **Inspection Complete!**")

                    if pred_class.lower() == 'defective':
                        st.error(f"## ❌ UNIT IS DEFECTIVE")
                    else:
                        st.success(f"## ✅ UNIT IS GOOD")

                    col_img, col_metrics = st.columns([1, 2])

                    with col_img:
                        st.image(temp_image_path, caption=uploaded_image.name, use_column_width=True)

                    with col_metrics:
                        st.metric(
                            label="Predicted Class",
                            value=pred_class.upper(),
                            delta=f"Confidence: {confidence * 100:.2f}%" if isinstance(confidence,
                                                                                       (int, float)) else None
                        )
                        st.text(f"Model File Used: {uploaded_model_file.name}")

                else:
                    st.error(f"❌ **Inspection Failed:** {inspection_result['message']}")
                    st.code(process.stderr, language='text')

            except Exception as e:
                st.error("❌ **An unexpected error occurred during inspection.**")
                st.exception(e)

            finally:
                # --- Cleanup ---
                if os.path.exists(TEMP_INSPECT_DIR):
                    shutil.rmtree(TEMP_INSPECT_DIR)