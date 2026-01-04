import yaml
import cv2
import os
import numpy as np
from typing import Union
from tensorflow.data import Dataset


def config_loader(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def frames_extraction(video_path, sequence_length=20, image_height=64, image_width=64):
    """
    Efficiently extract evenly spaced frames from a video, resize and normalize them.
    
    Args:
        video_path: Path to the video file.
        sequence_length: Number of frames to extract.
        image_height: Height to resize frames.
        image_width: Width to resize frames.
        
    Returns:
        frames_array: NumPy array of shape (sequence_length, image_height, image_width, 3)
                      with values normalized between 0 and 1.
    """

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    # Total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return np.zeros((sequence_length, image_height, image_width, 3), dtype=np.float32)

    # Compute indices of frames to extract (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

    # Preallocate array
    frames_array = np.zeros((sequence_length, image_height, image_width, 3), dtype=np.float32)

    current_frame = 0
    next_index = 0

    while next_index < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame == frame_indices[next_index]:
            # Resize and normalize
            resized = cv2.resize(frame, (image_width, image_height))  # correct order
            frames_array[next_index] = resized.astype(np.float32) / 255.0
            next_index += 1

        current_frame += 1

    # If video too short, repeat last frame
    while next_index < sequence_length:
        frames_array[next_index] = frames_array[next_index - 1]
        next_index += 1

    cap.release()
    return frames_array


def create_dataset(dataset_dir: str, classes_list: list, num_files: Union[int, None] = None, sequence_length: int = 20, seed: int = 42):
    '''
    This function will extract the data of the selected classes and create the required dataset.
    Returns:
        features:          A list containing the extracted frames of the videos.
        labels:            A list containing the indexes of the classes associated with the videos.
        video_files_paths: A list containing the paths of the videos in the disk.
    '''

    # Declared Empty Lists to store the features, labels and video file path values.
    features = []
    labels = []
    video_files_paths = []
    
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(classes_list):
        
        # Display the name of the class whose data is being extracted.
        print(f'Folder {class_index + 1}/{len(classes_list)}...Extracting Data of Class: {class_name}')
        
        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(dataset_dir, class_name))
        
        # Iterate through all the files present in the files list.
        np.random.seed(seed=seed)
        if num_files:
            files_list = np.random.permutation(files_list)[:num_files]
        else:
            files_list = np.random.permutation(files_list)
            
        for file_name in files_list:
            
            # Get the complete video path.
            video_file_path = os.path.join(dataset_dir, class_name, file_name)

            # Extract the frames of the video file.
            frames = frames_extraction(video_file_path)

            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.
            if len(frames) == sequence_length:

                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    # Converting the list to numpy arrays
    features = np.asarray(features) # type: ignore
    labels = np.array(labels)  
    
    dataset = Dataset.from_tensor_slices(features, labels)

    return features, labels