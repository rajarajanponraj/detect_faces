import os
import shutil
import logging
from deepface import DeepFace
import pandas as pd

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_faces(input_dir, output_dir, model_name="Facenet"):
    logging.info(f"Starting face detection with model {model_name}.")
    
    try:
        logging.info(f"Finding faces in images from {input_dir}.")
        
        # Loop through all the images in the input directory
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            
            # Only process image files
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                logging.info(f"Processing image: {img_path}")
                
                # Run DeepFace's detectFace method to detect faces in the image
                try:
                    face_image = DeepFace.detectFace(img_path=img_path, model_name=model_name, enforce_detection=False)
                    detected_face_path = os.path.join(output_dir, "detected_faces", img_name)
                    
                    # Save the detected face image
                    if not os.path.exists(os.path.join(output_dir, "detected_faces")):
                        os.makedirs(os.path.join(output_dir, "detected_faces"))
                    
                    logging.info(f"Saving processed face image to {detected_face_path}")
                    face_image.save(detected_face_path)
                except Exception as e:
                    logging.error(f"Error during face detection for {img_name}: {e}")

    except Exception as e:
        logging.error(f"Error during face finding: {e}")
