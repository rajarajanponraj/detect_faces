
# code to extract the faces from the images

# import os
# import logging
# import cv2
# import numpy as np
# from sklearn.cluster import DBSCAN

# # Set up logging configuration
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Function to detect faces in an image
# def detect_faces(input_dir):
#     # Load the pre-trained face detection model (using OpenCV Haar Cascade for simplicity)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # List to store the face embeddings (or coordinates)
#     faces = []

#     # Iterate over all images in the input directory
#     for image_file in os.listdir(input_dir):
#         image_path = os.path.join(input_dir, image_file)

#         if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             continue

#         # Read the image
#         image = cv2.imread(image_path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the image
#         detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in detected_faces:
#             # Crop the detected face from the image
#             face = image[y:y + h, x:x + w]
#             # For simplicity, we'll use the flattened pixel values of the face as an embedding
#             face_embedding = cv2.resize(face, (160, 160)).flatten()
#             faces.append(face_embedding)

#     logging.info(f"Detected {len(faces)} faces.")
#     return np.array(faces)

# # Function to cluster the detected faces
# def cluster_faces(faces, output_dir):
#     if faces.size == 0:
#         logging.info("No faces to cluster.")
#         return

#     # Apply DBSCAN clustering to the faces
#     clustering = DBSCAN(eps=0.5, min_samples=2).fit(faces)

#     # Create an output directory for each cluster
#     for cluster_id in set(clustering.labels_):
#         cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
#         os.makedirs(cluster_dir, exist_ok=True)

#         # Save images into their corresponding cluster directories
#         for i, label in enumerate(clustering.labels_):
#             if label == cluster_id:
#                 image_file = os.path.join(output_dir, f"image_{i}.jpg")
#                 face = faces[i].reshape((160, 160, 3))
#                 cv2.imwrite(image_file, face)

#     logging.info("Clustering completed and images saved.")

# # Main function to process images
# def process_images(input_dir, output_dir):
#     logging.info(f"Processing images from {input_dir} and saving clusters to {output_dir}")
    
#     # Ensure input directory exists
#     if not os.path.exists(input_dir):
#         logging.error(f"Input directory {input_dir} does not exist.")
#         return

#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Step 1: Detect faces in images
#     faces = detect_faces(input_dir)

#     # Step 2: Cluster the detected faces
#     cluster_faces(faces, output_dir)

# if __name__ == "__main__":
#     input_dir = "images"  # Directory containing images to process
#     output_dir = "output/clusters"  # Directory where clustered images will be saved
#     process_images(input_dir, output_dir)


# code to extract the faces from the images and split into folders

# import os
# import logging
# import cv2
# import numpy as np
# from sklearn.cluster import DBSCAN

# # Set up logging configuration
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Function to detect faces and extract them from images
# def detect_faces(input_dir):
#     # Load the pre-trained face detection model (using OpenCV Haar Cascade)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # List to store the face embeddings and original cropped face images
#     face_embeddings = []
#     face_images = []
#     face_file_names = []

#     # Iterate over all images in the input directory
#     for image_file in os.listdir(input_dir):
#         image_path = os.path.join(input_dir, image_file)

#         if not image_path.lower().endswith(('.png', '.jpg', '.jpeg','.webp')):
#             continue

#         # Read the image
#         image = cv2.imread(image_path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the image
#         detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in detected_faces:
#             # Crop the detected face from the image
#             face = image[y:y + h, x:x + w]
#             # Resize the face to a fixed size (160x160)
#             resized_face = cv2.resize(face, (160, 160))
#             # Flatten the resized face for clustering
#             face_embedding = resized_face.flatten()
#             # Append the cropped face, its embedding, and file name
#             face_images.append(resized_face)
#             face_embeddings.append(face_embedding)
#             face_file_names.append(image_file)

#     logging.info(f"Detected {len(face_embeddings)} faces.")
#     return np.array(face_embeddings), face_images, face_file_names

# # Function to cluster the detected faces and save them in separate folders
# def cluster_faces(face_embeddings, face_images, face_file_names, output_dir):
#     if len(face_embeddings) == 0:
#         logging.info("No faces to cluster.")
#         return

#     # Apply DBSCAN clustering to the face embeddings
#     clustering = DBSCAN(eps=30, min_samples=1, metric='euclidean').fit(face_embeddings)

#     # Create directories for clusters and save images
#     for cluster_id in set(clustering.labels_):
#         cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
#         os.makedirs(cluster_dir, exist_ok=True)

#         for i, label in enumerate(clustering.labels_):
#             if label == cluster_id:
#                 # Save the corresponding face image in the cluster directory
#                 output_path = os.path.join(cluster_dir, face_file_names[i])
#                 cv2.imwrite(output_path, face_images[i])

#     logging.info(f"Clustering completed. Clusters saved in {output_dir}.")

# # Main function to process images
# def process_images(input_dir, output_dir):
#     logging.info(f"Processing images from {input_dir} and saving clusters to {output_dir}")

#     # Ensure input directory exists
#     if not os.path.exists(input_dir):
#         logging.error(f"Input directory {input_dir} does not exist.")
#         return

#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Step 1: Detect and extract faces from images
#     face_embeddings, face_images, face_file_names = detect_faces(input_dir)

#     # Step 2: Cluster the extracted faces and save them
#     cluster_faces(face_embeddings, face_images, face_file_names, output_dir)

# if __name__ == "__main__":
#     input_dir = "images"  # Directory containing images to process
#     output_dir = "output/clusters"  # Directory where clustered faces will be saved
#     process_images(input_dir, output_dir)




import os
import shutil
import logging
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to detect faces and extract embeddings
def detect_faces(input_dir):
    # Load the pre-trained face detection model (using OpenCV Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Lists to store embeddings, cropped faces, and file paths
    face_embeddings = []
    face_images = []
    face_file_paths = []

    # Iterate over all images in the input directory
    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)

        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Read the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in detected_faces:
            # Crop the detected face from the image
            face = image[y:y + h, x:x + w]
            # Resize the face to a fixed size (160x160)
            resized_face = cv2.resize(face, (160, 160))
            # Flatten the resized face for clustering
            face_embedding = resized_face.flatten()

            # Append the cropped face, its embedding, and the original file path
            face_images.append(resized_face)
            face_embeddings.append(face_embedding)
            face_file_paths.append(image_path)

    logging.info(f"Detected {len(face_embeddings)} faces.")
    return np.array(face_embeddings), face_images, face_file_paths

# Function to cluster the detected faces and organize images
def cluster_faces(face_embeddings, face_images, face_file_paths, output_dir):
    if len(face_embeddings) == 0:
        logging.info("No faces to cluster.")
        return

    # Apply DBSCAN clustering to the face embeddings
    clustering = DBSCAN(eps=30, min_samples=1, metric='euclidean').fit(face_embeddings)

    # Create directories for clusters and save faces and original images
    for cluster_id in set(clustering.labels_):
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        for i, label in enumerate(clustering.labels_):
            if label == cluster_id:
                # Save the extracted face in the cluster directory
                face_output_path = os.path.join(cluster_dir, f"face_{i}.jpg")
                cv2.imwrite(face_output_path, face_images[i])

                # Copy the original image into the same cluster directory
                original_image_path = face_file_paths[i]
                shutil.copy(original_image_path, cluster_dir)

    logging.info(f"Clustering completed. Faces and images saved in {output_dir}.")

# Main function to process images
def process_images(input_dir, output_dir):
    logging.info(f"Processing images from {input_dir} and saving clusters to {output_dir}")

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        logging.error(f"Input directory {input_dir} does not exist.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Detect and extract faces from images
    face_embeddings, face_images, face_file_paths = detect_faces(input_dir)

    # Step 2: Cluster the extracted faces and organize input images
    cluster_faces(face_embeddings, face_images, face_file_paths, output_dir)

if __name__ == "__main__":
    input_dir = "images"  # Directory containing images to process
    output_dir = "output/clusters"  # Directory where clustered faces and images will be saved
    process_images(input_dir, output_dir)
