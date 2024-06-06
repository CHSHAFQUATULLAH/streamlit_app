import streamlit as st
from PIL import Image
import feedparser
import numpy as np
import cv2
import os
import torch
from torchvision import transforms
import base64
from ultralytics import YOLO

def fetch_news():
    agriculture_feeds = [
        "https://www.agweb.com/rss/news",
        "https://www.agriculture.com/rss/news",
        "https://www.farmprogress.com/rss.xml",
        "https://www.bbc.com/urdu/topics/c7zp5pqp86nt",
        "https://www.urdupoint.com/daily/live/agriculture.html",
        "https://www.brecorder.com/trends/agriculture",
        # Add more RSS feed URLs here
    ]

    all_news = []
    for feed_url in agriculture_feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            news_item = {
                "title": entry.title,
                "link": entry.link,
                "published": entry.published,
                # Use get() to handle missing 'summary' attribute
                "summary": entry.get('summary', '')
            }
            all_news.append(news_item)
    return all_news

def fetch_diseases():
    diseases = [
        {"name": "Coffee_healthy", "image_path": "https://github.com/CHSHAFQUATULLAH/streamlit_app/blob/851f72ea14175fbc175459502acc3923139c4f8b/images.jpeg", "info": [
            "Symptoms: No visible symptoms; plant appears healthy.", "Best-suited Medicine: No treatment required."]}
        {"name": "Cucumber_Blight", "image_path": "https://github.com/CHSHAFQUATULLAH/streamlit_app/blob/851f72ea14175fbc175459502acc3923139c4f8b/images.jpeg", "info": [
            "Symptoms: Dark, water-soaked lesions on leaves, stems, and fruit, leading to wilting and death of plants.", "Best-suited Medicine: Fungicides containing chlorothalonil or copper."]},
        # Add more diseases with their image paths and information here
    ]
    return diseases

def crop_disease_detection_app():
    logo_path = 'https://github.com/CHSHAFQUATULLAH/streamlit_app/blob/851f72ea14175fbc175459502acc3923139c4f8b/images.jpeg'

    try:
        logo = Image.open(logo_path)
        # Set the desired size (width, height) for the logo
        desired_size = (20, 10)

        st.sidebar.image(logo, use_column_width=True)

    except Exception as e:
        st.sidebar.write(f"Error loading logo: {e}")

    # Set up the sidebar for navigation
    st.sidebar.title("Menu")
    menu_options = ["Home", "News", "Disease Library",
                    "Detect Disease", "Contact",  "About Us"]
    button_styles = {
        "Home": "primary",
        "Detect Disease": "primary",
        "News": "primary",
        "Disease Library": "primary",
        "Contact": "primary",
        # "Services": "primary",
        "About Us": "primary"
    }

    selected_option = st.sidebar.radio("Navigation", menu_options, index=0)

    if selected_option == "Home":
        st.title("Automated Crop Disease Detection")
        st.markdown("""
        Crop disease detection is essential for ensuring food security and maximizing agricultural yield. 
        Our web-based application leverages artificial intelligence to identify and diagnose diseases in crops, enabling farmers to take timely actions and mitigate crop losses.
        """)

        st.subheader("The Challenge")
        st.write("Crop diseases can devastate entire harvests if not detected and treated early. However, farmers face challenges in identifying diseases accurately and promptly, leading to significant economic losses.")

        img2 = Image.open('https://github.com/CHSHAFQUATULLAH/streamlit_app/blob/851f72ea14175fbc175459502acc3923139c4f8b/images.jpeg')
        st.image(img2, use_column_width=True)

        st.subheader("Why Choose Automated Crop Disease Detection(ACDD)")
        st.markdown("""
        - **Early Detection:** Identifies crop diseases at an early stage, minimizing crop losses.
        - **Accurate Diagnosis:** Utilizes advanced AI algorithms to provide accurate disease diagnoses.
        - **User-Friendly Interface:** Simple and intuitive interface accessible to farmers of all levels.
        - **Timely Interventions:** Enables farmers to take timely actions to prevent the spread of diseases.
        """)

        st.subheader("Services we provide")
        st.markdown("""
        - **Agriultural News:**  Updated news about agriulture are displayed up to market trends.
        - **Disease Library**   A precise and abstrat overview of disease itrs symptoms and ure .
        - **Detect Disease**    A three step disease identifiation.
        - **Contat Us**         If you are facing any bugs ort isssues related to our app you an contat us on given details.
        
        """)
                
        st.subheader("How to use ACDD(3 Step Diagnosis)")
        st.markdown("""
        - **Step 1:**  Goto detect disease page and clik on browse image and select image of crop you want to chek.
        - **Step 2**   Click on predict button and wait for a few seconds.
        - **Step 3**   The disease found ìn your uploaded image will be displayed.
        
        """)
        

    elif selected_option == "News":
        st.title("Agriculture News")
        st.write("Stay updated with the latest agriculture news.")
        news_items = fetch_news()
        for item in news_items:
            st.subheader(item['title'])
            st.write(item['summary'])
            st.write(f"[Read more]({item['link']})")

    elif selected_option == "Disease Library":
        st.title("Disease Library")
        st.write("""
         Our app is compatible with six crops cucumber, coffee, cotton, olive, tomato, wheat
         
         An abstract info about the diseases in these crops their symptoms and cure is given below:
         """)
        
        diseases = fetch_diseases()
        for disease in diseases:
            st.subheader(disease["name"])
            try:
                img = Image.open(disease["image_path"])
                st.image(img, use_column_width=True)
            except Exception as e:
                st.write(f"Error loading image for {disease['name']}: {e}")
            st.write(", ".join(disease["info"]))

    elif selected_option == "Contact":
        st.title("Contact Us")
        st.write("""
         We are here to assist you with any inquiries or support you may need. Please feel free to reach out to us via email or phone, and we'll be happy to assist you.
         """)

        st.header("Supervisor:")
        st.subheader("Dr. Nouman Noor")
        st.write("""
         Dr. Nouman Noor is the project supervisor and an expert in deep learning technology. He oversees the development and implementation of innovative solutions to enhance crop yield and disease management.
         """)
        st.write(
            "**Email:** [drnoumannoor@example.com](mailto:nouman@example.com)")
        st.write("**Phone:** [+92 123-4567890]")

        st.header("Students:")

        st.subheader("Ch Shafquatullah")
        st.write("""
         Ch Shafquatullah is a dedicated member of our team, specializing in IoT and agricultural technology. He has played a key role in developing the crop disease detection application, leveraging his expertise to create user-friendly and efficient solutions.
         """)
        st.write(
            "**Email:** [chshafquatullahn@gmail.com](mailto:chshafquatullahn@gmail.com)")
        st.write("**Phone:** [+92 336-4535423]")

        st.subheader("Masab")
        st.write("""
         Masab is a talented student with a passion for agriculture and technology. He has contributed to the project by conducting research and implementing innovative features to enhance the functionality of the crop disease detection application.
         """)
        st.write("**Email:** [masab@example.com](mailto:masab@example.com)")
        st.write("**Phone:** [+92 123-4567890]")

        st.subheader("Muzammil Hussain")
        st.write("""
         Muzammil is a motivated team member with a strong background in machine learning and computer vision. He has played a crucial role in developing the AI algorithms used for disease detection in crops, ensuring accurate and reliable results.
         """)
        st.write(
            "**Email:** [muzammil@example.com](mailto:muzammil@example.com)")
        st.write("**Phone:** [+92 123-4567890]")

        st.write("""
         We value your feedback and are committed to providing exceptional service to our customers. Don't hesitate to contact us with any questions or concerns.
         """)
    elif selected_option == "About Us":
        st.title("About Us")
        st.write("""
        Our team is composed of passionate experts dedicated to revolutionizing agriculture through the power of technology. With a deep understanding of the challenges faced by farmers worldwide, we have embarked on a mission to leverage the Internet of Things (IoT) to transform farming practices.

        At our core, we believe in the potential of IoT to empower farmers with real-time data, actionable insights, and automated solutions. By integrating sensors, actuators, and intelligent algorithms, we aim to create smart agricultural systems that optimize resource usage, enhance productivity, and minimize environmental impact.

        With a focus on sustainability and scalability, we are committed to developing IoT solutions that are accessible to farmers of all scales and backgrounds. Through our collaborative efforts with agricultural communities, research institutions, and technology partners, we strive to foster innovation and drive positive change in the agricultural sector.

        Our vision is to build a future where technology enables farmers to overcome challenges, maximize yields, and ensure food security for generations to come. By harnessing the potential of IoT, we aim to create a more resilient and sustainable food system that benefits farmers, consumers, and the planet.

        Join us on our journey as we work towards a future where technology and agriculture converge to shape a better world.
        """)
    elif selected_option == "Detect Disease":

        BASE_DIR = r'https://github.com/CHSHAFQUATULLAH/streamlit_app/blob/212f09a14e8a5f1b7f3ad2b2977adc66a83015c6/'
        MODEL_NAME = 'best (2).pt'
        MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
        LABELS_PATH = r'https://github.com/CHSHAFQUATULLAH/streamlit_app/blob/02e9d30d2170736bdb1d32369691d624186b2b9a/labels.txt'

        def load_yolo_model(model_path):
            try:
                # Load the YOLOv8 model
                model = YOLO(model_path)
                return model
            except Exception as e:
                st.error(
                    f"An error occurred while loading the model: {str(e)}")
                return None

        def load_labels(labels_path):
            try:
                with open(labels_path, 'r') as file:
                    labels = file.read().splitlines()
                return labels
            except Exception as e:
                st.error(
                    f"An error occurred while loading the labels: {str(e)}")
                return None

        def preprocess_image(image):
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ])
            return transform(image).unsqueeze(0)

        def detect_objects_yolo(image, model):
            image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            results = model(image_cv2)

            return results

        def draw_bounding_boxes(image, results, labels, confidence_threshold=0.5):

            predictions = results[0].boxes

            for pred in predictions:
                confidence = pred.conf.item()
                if confidence > confidence_threshold:
                    x1, y1, x2, y2 = map(int, pred.xyxy[0].tolist())
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    class_id = int(pred.cls)
                    label = f"{labels[class_id]}: {confidence:.2f}" if class_id < len(
                        labels) else f"Unknown: {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    st.write("The diseases found are:", label)
                    
                

            return image
        st.write("Kindly check Disease Library to check possible cure")


        # Main function for the Streamlit app
        def app():
            st.title("Detect Disease")
            st.write("""
        Just upload the image of your crop and we will be happyy to let you know what disease it has 
        
        You can also check our Library page for a precise description about the disease and the best
        
        suited medicines for it.
        """)

            # Load model
            model = load_yolo_model(MODEL_PATH)

            if model is not None:
                st.success("Model loaded successfully!")

            # Load labels
            labels = load_labels(LABELS_PATH)

            if labels is not None:
                st.success("Labels loaded successfully!")

            # Upload file option
            st.markdown(
                '<h1 style="color:white;">Upload Image for Object Detection</h1>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose a file", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
                # Read the uploaded image
                image = Image.open(uploaded_file)

                # Display the uploaded image
                st.image(image, caption="Uploaded Image",
                         use_column_width=True)

                # Check if the "Predict" button is clicked
                if st.button("Predict"):
                    # Perform object detection if model loaded successfully
                    if model is not None and labels is not None:
                        results = detect_objects_yolo(image, model)
                        if results is not None:
                            # Convert image to array format for drawing bounding boxes
                            image_np = np.array(image)
                            # Draw bounding boxes on the image
                            image_with_boxes = draw_bounding_boxes(
                                image_np, results, labels)

                            # Display the image with detected objects
                            st.image(
                                image_with_boxes, caption="Detected Objects", use_column_width=True)
                            
                       
                        else:   
                            st.error(
                                "Object detection could not be performed due to an error.")
                    else:
                        st.error(
                            "Model or labels not loaded. Please check the model file path and labels file path and try again.")
                        
        app()


# Run the crop disease detection app
if __name__ == "__main__":
    crop_disease_detection_app()
