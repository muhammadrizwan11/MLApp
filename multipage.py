import streamlit as st
from PIL import Image

# Function for Object Detection (Replace this with your actual object detection code)
def object_detection():
    
    # Your object detection code here
    
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.utils import draw_bounding_boxes
    import zipfile
    import os

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    categories = weights.meta["categories"]
    img_preprocess = weights.transforms()

    # Function to load the model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

    def load_model():
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        model.eval()
        return model

    model = load_model()

    # Function to make predictions
    def make_prediction(img):
        img_processed = img_preprocess(img)
        prediction = model(img_processed.unsqueeze(0))[0]
        prediction["labels"] = [categories[label] for label in prediction["labels"]]
        return prediction

    # Function to create image with bounding boxes
    def create_image_with_bboxes(img, prediction):
        img_tensor = torch.tensor(img)
        img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"],
                                          labels=prediction["labels"],
                                          colors=["red" if label == "person" else "green" for label in
                                                  prediction["labels"]], width=2)
        img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
        return img_with_bboxes_np

    # Function to handle dataset upload and extract images
    def handle_dataset_upload(zip_file):
        images = []

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract the contents of the ZIP file to a temporary directory
            temp_dir = "temp_dataset"
            zip_ref.extractall(temp_dir)

            # Get a list of image files in the extracted directory
            image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Load each image and append to the images list
            for image_file in image_files:
                image_path = os.path.join(temp_dir, image_file)
                image = Image.open(image_path)
                images.append(image)

        return images

    # Dashboard
    st.markdown("<h1 style='text-align: center;'>Object Detector Model</h1>", unsafe_allow_html=True)


    # Step 1: Upload ZIP Dataset
    upload_dataset = st.file_uploader(label="Upload Dataset (ZIP file format):", type=["zip"])

    if upload_dataset:
        # Handle dataset upload and extract images
        images = handle_dataset_upload(upload_dataset)

        # Display the extracted images and predictions
        for image in images:
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Make predictions and display bounding boxes
            image = image.convert('RGB')
            prediction = make_prediction(image)
            img_with_bbox = create_image_with_bboxes(np.array(image).transpose(2, 0, 1), prediction)

            # Display the image with bounding boxes
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            plt.imshow(img_with_bbox)
            plt.xticks([], [])
            plt.yticks([], [])
            ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)

            # Display predicted probabilities
            del prediction["boxes"]
            st.header("Predicted Probabilities")
            st.write(prediction, style={"labels": {"font-size": "20px"}})

        
        # Button to export the trained model
        # if st.button("Export Trained Model"):
        #     # Save the model to a file
        #     model_filename = "trained_model.pth"
        #     torch.save(model.state_dict(), model_filename)
        #     st.success(f"Trained model exported successfully as {model_filename}")


    

# Function for Image Classification (Replace this with your actual image classification code)
def image_classifier():
    
    # Your image classification code here    
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from torchvision.models import resnet50, ResNet50_Weights
    from captum.attr import IntegratedGradients
    import torch
    import zipfile

    preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
    categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta['categories'])

    @st.cache_resource
    def load_model():
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
        return torch.jit.script(model)  # Convert the model to TorchScript

    model = load_model()

    def make_prediction(model, processed_img):
        probs = model(processed_img.unsqueeze(0))
        probs = probs.softmax(1)
        probs = probs[0].detach().numpy()
        prob, idxs = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
        return prob, idxs

    def interpret_prediction(model, processed_img, target):
        interpretation_algo = IntegratedGradients(model)
        feature_imp = interpretation_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    
        # Squeeze to match the size
        feature_imp = feature_imp.squeeze(0).numpy()
        feature_imp = feature_imp.transpose(1, 2, 0)
        return feature_imp

    # def save_model(model, file_path):
    #     torch.save(model.state_dict(), file_path)
    #     st.success("Model exported successfully!")



    # Dashboard******************************************************
    
    st.markdown("<h1 style='text-align: center;'>Image classifier Model</h1>", unsafe_allow_html=True)

    

    upload = st.file_uploader(label="Upload zip file:", type=["zip"])

    if upload:
        # Extract images from the zip file
        with zipfile.ZipFile(upload, 'r') as zip_ref:
            zip_ref.extractall('temp_images')

        # Train/Predict Model Button
        if st.button("Train/Predict Model"):
            # Display and process each image
            predictions = []
            image_files = zip_ref.namelist()
            for image_file in image_files:
                img_path = f'temp_images/{image_file}'
                img = Image.open(img_path).convert('RGB')

                st.image(img, caption=f"Uploaded image: {image_file}", use_column_width=True)

                model = load_model()
                processed_img = preprocess_func(img)
                probs, idxs = make_prediction(model, processed_img)
                predictions.append(idxs[0])  # Use the top predicted class index

                # Visualization of predictions
                main_fig = plt.figure(figsize=(12, 3))
                ax = main_fig.add_subplot(111)
                plt.barh(y=categories[idxs][::-1], width=probs[::-1], color="tomato")
                plt.title("Top 5 Possibilities", loc="center", fontsize=15)
                st.pyplot(main_fig, use_container_width=True)

            # Calculate and print accuracy
            consistency = len(set(predictions)) / len(predictions)
            accuracy_text = f"Accuracy Of Model Is: {consistency * 100:.2f}%"
            # Display accuracy with green border
            st.markdown(f'<p style="text-align:center; border: 2px solid green; padding: 10px;">{accuracy_text}</p>', unsafe_allow_html=True)

            # Button to export the trained model
        # if st.button("Export Trained Model"):
        #     # Save the model to a file
        #     model_filename = "trained_model.pth"
        #     torch.save(model.state_dict(), model_filename)
        #     st.success(f"Trained model exported successfully as {model_filename}")


            # Clean up temporary images folder
            zip_ref.close()
            import shutil
            shutil.rmtree('temp_images', ignore_errors=True)




# Function for Creating Graph (Replace this with your actual graph creation code)
def create_graph():
    # Your graph creation code here
    
    import pandas as pd
    import zipfile
    import io
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shutil

    def extract_zip_file(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as z:
            # Extract all files to a temporary directory
            temp_dir = 'temp_extracted_files'
            z.extractall(temp_dir)
            # Return the list of extracted files
            return [file for file in z.namelist()]

    def main():
        st.markdown("<h1 style='text-align: center;'>Graph Prediction Model</h1>", unsafe_allow_html=True)

        # Instructions for users
        st.markdown("<h2>Instructions for Users</h2>", unsafe_allow_html=True)

        st.markdown(
        """

        * Upload a ZIP file containing one or more CSV or Excel files.
        * Once uploaded, the contents of the ZIP file will be extracted.
        * Select one of the extracted files to visualize its data.
        * On the sidebar, select the columns you want to visualize by clicking on the checkboxes.
        * Choose the plot type (Line Plot, Bar Chart, Scatter Plot, Pie Chart, Radar Chart) using the radio buttons.
        * You can also choose to display the entire dataset plot by checking the "Show Entire Dataset" checkbox.
        * Click on the buttons or checkboxes to generate the corresponding graphs.
        """
        )

        # User input: Upload ZIP file
        zip_file = st.file_uploader("Upload a ZIP file", type=["zip"])

        # Process uploaded ZIP file
        if zip_file is not None:
            try:
                # Extract ZIP file
                extracted_files = extract_zip_file(zip_file)

                # Display the list of extracted files
                selected_file = st.selectbox("Select a file to visualize:", extracted_files)

                # Determine file type and read into a DataFrame accordingly
                if selected_file.endswith('.xlsx'):
                    df = pd.read_excel(os.path.join('temp_extracted_files', selected_file))
                elif selected_file.endswith('.csv'):
                    df = pd.read_csv(os.path.join('temp_extracted_files', selected_file))

                # Display the raw data
                st.subheader("Raw Data:")
                st.write(df)

                # Display buttons for each column in the sidebar
                with st.sidebar:
                    st.subheader("Select Columns:")
                    column_buttons = st.multiselect("", df.columns)
                    plot_type = st.radio("Choose Plot Type:", ["Line Plot", "Bar Chart", "Scatter Plot", "Pie Chart", "Radar Chart"])
                    show_entire_dataset = st.checkbox("Show Entire Dataset")

                # Plotting based on selected columns and plot type
                if show_entire_dataset:
                    st.subheader(f"Graph for Entire Dataset ({plot_type}):")
                    if plot_type == "Line Plot":
                        # Create line chart for the entire dataset
                        fig, ax = plt.subplots()
                        df.plot(ax=ax)
                        st.pyplot(fig)
                    elif plot_type == "Bar Chart":
                        # Create bar chart for the entire dataset
                        fig, ax = plt.subplots()
                        df.plot(kind='bar', ax=ax)
                        st.pyplot(fig)
                    elif plot_type == "Scatter Plot":
                        # Create scatter plot for the entire dataset
                        fig, ax = plt.subplots()
                        sns.scatterplot(data=df, ax=ax)
                        st.pyplot(fig)
                    elif plot_type == "Pie Chart":
                        # Create pie chart for the entire dataset
                        fig, ax = plt.subplots()
                        df.sum().plot.pie(autopct=lambda p: '{:.1f}%'.format(p) if isinstance(p, (int, float)) else '', ax=ax)
                        st.pyplot(fig)
                    elif plot_type == "Radar Chart":
                        # Create radar chart for the entire dataset
                        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
                        for column in df.columns:
                            ax.plot(df.columns, df.loc[0], label=column)
                        ax.legend()
                        st.pyplot(fig)

                # Plotting based on selected columns and plot type
                if column_buttons:
                    for column in column_buttons:
                        st.subheader(f"Graph for {column} ({plot_type}):")
                        if plot_type == "Line Plot":
                            if df[column].dtype in ['int64', 'float64']:
                                # Create line chart for numeric columns
                                fig, ax = plt.subplots()
                                sns.lineplot(x=df.index, y=df[column], ax=ax)
                                st.pyplot(fig)
                            else:
                                st.warning("Line plots are suitable for numeric columns only.")
                        elif plot_type == "Bar Chart":
                            if df[column].dtype in ['int64', 'float64']:
                                # Create bar chart for numeric columns
                                fig, ax = plt.subplots()
                                sns.barplot(x=df.index, y=df[column], ax=ax)
                                st.pyplot(fig)
                            else:
                                # Create count plot for categorical columns
                                fig, ax = plt.subplots()
                                sns.countplot(x=column, data=df, ax=ax)
                                st.pyplot(fig)
                        elif plot_type == "Scatter Plot":
                            # Create scatter plot for numeric columns
                            fig, ax = plt.subplots()
                            sns.scatterplot(x=df.index, y=df[column], ax=ax)
                            st.pyplot(fig)
                        elif plot_type == "Pie Chart":
                            # Create pie chart for numeric columns
                            fig, ax = plt.subplots()
                            df[column].plot.pie(autopct=lambda p: '{:.1f}%'.format(p) if isinstance(p, (int, float)) else '', ax=ax)
                            st.pyplot(fig)
                        elif plot_type == "Radar Chart":
                            # Create radar chart for numeric columns
                            fig, ax = plt.subplots(subplot_kw=dict(polar=True))
                            ax.plot(df.columns, df.loc[0])
                            ax.fill(df.columns, df.loc[0], alpha=0.25)
                            st.pyplot(fig)

                # Cleanup: Delete temporary extracted files
                for file in extracted_files:
                    os.remove(os.path.join('temp_extracted_files', file))
                shutil.rmtree('temp_extracted_files', ignore_errors=True)

            except pd.errors.EmptyDataError:
                st.error("The uploaded ZIP file is empty or does not contain valid CSV or Excel files.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    if __name__ == "__main__":
     main()



















# Main Streamlit App
def main():
    #st.title("Machine Learning Application")
    st.markdown("<h1 style='text-align: center;'>Machine Learning Application</h1>", unsafe_allow_html=True)

    # Sidebar with navigation buttons
    selected_page = st.sidebar.selectbox("Select Page", ["Home", "Object Detection", "Image Classifier", "Create Graph"])

    # Page content based on selection
    if selected_page == "Home":
        st.write("Welcome to the Home Page")
    elif selected_page == "Object Detection":
        object_detection()
    elif selected_page == "Image Classifier":
        image_classifier()
    elif selected_page == "Create Graph":
        create_graph()

# Run the app
if __name__ == "__main__":
    main()
