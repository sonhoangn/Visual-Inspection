import streamlit as st
import os
import subprocess
import json
# Removed unused 'shutil' import

# --- Configuration for Streamlit App ---
st.set_page_config(layout="wide", page_title="Model Training & Inspection App")
st.title("Image Classification Web App")
st.write("---")

# Initialize session state variables for persistence across tabs
if 'model_save_path' not in st.session_state:
    st.session_state['model_save_path'] = "my_trained_model/defect_model.keras"
if 'training_result' not in st.session_state:
    st.session_state['training_result'] = None

# 1. Training Logic Wrapped in a Function

# @st.cache_resource is used for heavy-to-create objects like ML models.
# It makes sure the training is only run once unless the input parameters change.
@st.cache_resource(show_spinner="Training model... please wait! ⏳")
def train_model_function(data_dir, save_path, epochs, img_size, batch_size):
    """
    Runs the model training script as a subprocess and saves the model.

    In a real-world app, you would move the contents of 'train_model.py'
    into this function, but for integration, running the script is easier.
    """
    st.info(f"Starting training with data from: **{data_dir}**")

    # Define the environment variables for the subprocess to use
    env = os.environ.copy()
    env['DATA_DIR'] = data_dir
    env['MODEL_SAVE_PATH'] = save_path
    env['EPOCHS'] = str(epochs)  # Pass other params as needed

    try:
        # Run the training script as a separate Python process
        # We assume 'train_model.py' is in the same directory
        process = subprocess.run(
            ['python', 'train_model.py'],
            check=True,  # Raise an exception for non-zero exit codes
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,  # Pass the directory paths to the script via env vars
            text=True
        )

        # Display the output from the training script
        st.code(process.stdout, language='text')

        # Check if the model file was actually created
        if os.path.exists(save_path):
            st.success(f"✅ **Training complete!** Model saved to: `{save_path}`")
            return f"Model trained and saved at {save_path}"
        else:
            st.error("Training finished, but model file was not found. Check the script output.")
            return None

    except subprocess.CalledProcessError as e:
        st.error("❌ **Training failed!** Check the error log below.")
        st.exception(e)
        st.code(e.stderr, language='text')
        return None
    except FileNotFoundError:
        st.error("❌ **Error:** Could not find 'train_model.py' or 'python' interpreter.")
        return None

# Split App into Tabs
tab_train, tab_inspect = st.tabs(["🚀 Model Training", "🔍 Unit Inspection"])

# =========================================================================
# 🚀 MODEL TRAINING TAB (FIXED UI PLACEMENT)
# =========================================================================
with tab_train:
    st.header("1. Configure Training Parameters")

    # Use a form to group inputs and control when the app reruns
    with st.form("training_config_form"):
        col1, col2 = st.columns(2)

        with col1:
            data_dir_path = st.text_input(
                "📂 **Image Data Directory Path**",
                value="data/train",
                key="data_dir_input",
                help="Enter the local path to the folder containing your image class subdirectories (e.g., 'data/train')."
            )

        with col2:
            # Connect this input directly to the session state variable
            st.session_state['model_save_path'] = st.text_input(
                "💾 **Model Save Path**",
                value=st.session_state['model_save_path'],
                key="model_path_input",
                help="Enter the full path, including the filename, to save the trained model (e.g., 'models/my_model.keras')."
            )
            model_save_path_local = st.session_state['model_save_path'] # Local alias for use below

        # Advanced Configuration (Optional)
        with st.expander("Advanced Configuration"):
            epochs = st.slider("Number of Epochs", 1, 50, 10)
            batch_size = st.slider("Batch Size", 16, 128, 32)
            img_height = st.number_input("Image Height (px)", 32, 512, 128)
            img_width = st.number_input("Image Width (px)", 32, 512, 128)

        st.write("---")
        submitted = st.form_submit_button("🚀 **Start Model Training**", type="primary")

    if submitted:
        # 3.1. Basic Validation
        if not os.path.isdir(data_dir_path):
            st.error(f"⚠️ **Error:** Data directory not found at `{data_dir_path}`. Please check the path.")
        else:
            # 3.2. Ensure the save directory exists
            save_dir = os.path.dirname(model_save_path_local)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                st.warning(f"Created model save directory: `{save_dir}`")

            # 3.3. Call the training function
            st.session_state['training_result'] = train_model_function(
                data_dir_path,
                model_save_path_local,
                epochs,
                (img_height, img_width),
                batch_size
            )

    # 4. Display Final Result
    if st.session_state['training_result']:
        st.balloons()
        st.markdown(f"### 🎉 Success: {st.session_state['training_result']}")

# =========================================================================
# 🔍 UNIT INSPECTION TAB
# =========================================================================
with tab_inspect:
    st.header("1. Provide Input")

    # Read model path directly from session state
    inspection_model_path = st.text_input(
        "📂 **Trained Model Path (.keras)**",
        value=st.session_state['model_save_path'],
        help="Path to the saved Keras model file."
    )

    # File uploader for the image to inspect
    uploaded_image = st.file_uploader(
        "🖼️ **Upload Image for Inspection**",
        type=['jpg', 'jpeg', 'png']
    )

    st.write("---")
    st.header("2. Inspection Result")

    # Use a separate button for inspection
    if uploaded_image is not None and st.button("🔍 **Run Inspection**"):
        # ... (Your existing inspection button logic and display remains correct) ...
        # Temporary directory to save the uploaded image for the subprocess to access
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, uploaded_image.name)

        # Save the uploaded file to disk
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # --- Run the Inspection Subprocess ---
        with st.spinner("Inspecting unit..."):
            try:
                # Run the inspection script
                process = subprocess.run(
                    ['python', 'inspect_unit.py', temp_image_path, inspection_model_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # The output is expected to be a single JSON string
                result_json = process.stdout.strip()
                inspection_result = json.loads(result_json)

                # --- Display Results ---
                if inspection_result["success"]:
                    pred_class = inspection_result["prediction_class"]
                    confidence = inspection_result["confidence"]

                    st.success("✅ **Inspection Complete!**")

                    # Custom display based on prediction
                    if pred_class == 'defective':
                        st.error(f"## ❌ UNIT IS DEFECTIVE")
                        st.balloons()
                    else:
                        st.success(f"## ✅ UNIT IS GOOD")

                    col_img, col_metrics = st.columns([1, 2])

                    with col_img:
                        st.image(temp_image_path, caption=uploaded_image.name, use_column_width=True)

                    with col_metrics:
                        st.metric(
                            label="Predicted Class",
                            value=pred_class.upper(),
                            delta=f"Confidence: {confidence * 100:.2f}%"
                        )
                        st.metric(
                            label="Raw Score (P('good'))",
                            value=f"{inspection_result['raw_score_good']:.4f}"
                        )

                else:
                    st.error(f"❌ **Inspection Failed:** {inspection_result['message']}")
                    st.code(process.stderr, language='text')

            except subprocess.CalledProcessError as e:
                st.error("❌ **Script Error:** The inspection script terminated with an error.")
                st.code(e.stderr, language='text')
            except json.JSONDecodeError:
                st.error("❌ **Result Error:** Could not parse the script output.")
                st.text(process.stdout)
            finally:
                # Cleanup the temporary image file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)