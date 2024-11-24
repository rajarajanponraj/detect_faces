import os
import shutil
import logging
from deepface import DeepFace
import pandas as pd

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cluster_faces(input_dir, output_dir, model_name="Facenet", db_path="path_to_face_database"):
    logging.info(f"Starting clustering with model {model_name}.")
    
    try:
        logging.info(f"Finding faces in images from {input_dir}.")
        
        # Loop through all the images in the input directory
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            
            # Only process image files
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                logging.info(f"Processing image: {img_path}")
                
                # Run DeepFace's find method to recognize faces from the image and compare with the db_path
                try:
                    results = DeepFace.find(img_path=img_path, model_name=model_name, db_path=db_path, enforce_detection=False)
                    logging.info(f"Found {len(results)} results for {img_name}.")
                    
                    if isinstance(results, pd.DataFrame) and not results.empty:
                        # Create output directory for clusters if it doesn't exist
                        cluster_dir = os.path.join(output_dir, "clusters")
                        os.makedirs(cluster_dir, exist_ok=True)
                        
                        # Copy the image to the cluster directory
                        for img_path in results["identity"].tolist():
                            try:
                                logging.info(f"Copying image {img_path} to {cluster_dir}.")
                                shutil.copy(img_path, cluster_dir)
                            except Exception as e:
                                logging.error(f"Error copying image {img_path}: {e}")
                except Exception as e:
                    logging.error(f"Error during face finding for {img_name}: {e}")

    except Exception as e:
        logging.error(f"Error during face finding: {e}")
