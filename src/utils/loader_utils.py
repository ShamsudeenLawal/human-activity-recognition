import yaml
import cv2
import os
import numpy as np
from typing import Union
import tensorflow as tf
from sklearn.model_selection import train_test_split


def config_loader(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def frames_extraction(video_path: str, sequence_length: int = 20, image_height: int = 64, image_width: int = 64, channels: int = 3):
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
        return np.zeros((sequence_length, image_height, image_width, channels), dtype=np.float32)

    # Compute indices of frames to extract (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

    # Preallocate array
    frames_array = np.zeros((sequence_length, image_height, image_width, channels), dtype=np.float32)

    current_frame = 0
    next_index = 0

    while next_index < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame == frame_indices[next_index]:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed
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


def create_dataset(dataset_dir: str, classes_list: list[str], num_files: Union[int, None] = None,
                   sequence_length: int = 20, seed: int = 42, image_height: int = 64, image_width: int =64,
                   channels=3, test_size: float = 0.2):
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
    np.random.seed(seed=seed)
    for class_index, class_name in enumerate(classes_list):
        # Display the name of the class whose data is being extracted.
        print(f'Folder {class_index + 1}/{len(classes_list)}...Extracting Data of Class: {class_name}')
        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(dataset_dir, class_name))
        # Iterate through all the files present in the files list.
        files_list = np.random.permutation(files_list)
        if num_files is not None:
            files_list = files_list[:num_files]

        for file_name in files_list:
            
            # Get the complete video path.
            video_file_path = os.path.join(dataset_dir, class_name, file_name)

            # Extract the frames of the video file.
            frames = frames_extraction(video_path=video_file_path, sequence_length=sequence_length,
                                       image_height=image_height, image_width=image_width, channels=channels)

            # Append the data to their repective lists.
            features.append(frames)
            labels.append(class_index)
            video_files_paths.append(video_file_path)

    # Converting the list to numpy arrays
    features = np.asarray(features) # type: ignore
    labels = np.array(labels)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed
    )
    
    # create tensorflow train and test datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return train_dataset, test_dataset

# ============================================================================================

# Using the generator approach
def video_generator(
    dataset_dir: str,
    classes_list: list[str],
    sequence_length: int = 20,
    image_height: int = 64,
    image_width: int = 64,
    channels: int = 3,
    num_files: int | None = None,
    seed: int = 42
):
    np.random.seed(seed)

    for class_index, class_name in enumerate(classes_list):
        class_dir = os.path.join(dataset_dir, class_name)
        files_list = os.listdir(class_dir)
        files_list = np.random.permutation(files_list)

        if num_files is not None:
            files_list = files_list[:num_files]

        for file_name in files_list:
            video_path = os.path.join(class_dir, file_name)

            frames = frames_extraction(
                video_path=video_path,
                sequence_length=sequence_length,
                image_height=image_height,
                image_width=image_width,
                channels=channels,
            )

            yield frames, class_index


def get_video_paths(dataset_dir: str, classes_list: list[str]):
    video_paths = []
    labels = []

    for class_index, class_name in enumerate(classes_list):
        class_dir = os.path.join(dataset_dir, class_name)
        for file in os.listdir(class_dir):
            video_paths.append(os.path.join(class_dir, file))
            labels.append(class_index)

    return np.array(video_paths), np.array(labels)


def video_generator_from_paths(
    video_paths,
    labels,
    sequence_length,
    image_height,
    image_width,
    channels,
):
    for video_path, label in zip(video_paths, labels):
        frames = frames_extraction(
            video_path,
            sequence_length,
            image_height,
            image_width,
            channels,
        )
        yield frames, label


def make_dataset(video_paths, labels, sequence_length, image_height, image_width, channels):
    output_signature = (
        tf.TensorSpec(
            shape=(sequence_length, image_height, image_width, channels),
            dtype=tf.float32
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.int32
        ),
    )

    dataset = tf.data.Dataset.from_generator(
        video_generator_from_paths,
        args=(
            video_paths,
            labels,
            sequence_length, image_height, image_width, channels,
        ),
        output_signature=output_signature
    )

    return dataset


def create_dataset_from_generator(
    dataset_dir: str,
    classes_list: list[str],
    sequence_length: int = 20,
    image_height: int = 64,
    image_width: int = 64,
    channels: int = 3,
    num_files: int | None = None,
    seed: int = 42,
    test_size: float = 0.2,
):
    output_signature = (
        tf.TensorSpec(
            shape=(sequence_length, image_height, image_width, channels),
            dtype=tf.float32
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.int32
        )
    )

    video_paths, labels = get_video_paths(dataset_dir=dataset_dir, classes_list=classes_list)

    X_train, X_test, y_train, y_test = train_test_split(
                                        video_paths, labels, test_size=test_size,
                                        random_state=seed, stratify=labels
                                        )

    train_dataset = make_dataset(X_train, y_train, sequence_length, image_height, image_width, channels)
    test_dataset = make_dataset(X_test, y_test, sequence_length, image_height, image_width, channels)

    return train_dataset, test_dataset


# ============================================================================================

# def continuous_frames_extraction(
#     video_path: str,
#     sequence_length: int = 20,
#     image_height: int = 64,
#     image_width: int = 64,
#     channels: int = 3,
# ):
#     """
#     Extract the first `sequence_length` frames from a video,
#     resize and normalize them.
#     """

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video {video_path}")

#     frames_array = np.zeros(
#         (sequence_length, image_height, image_width, channels),
#         dtype=np.float32,
#     )

#     frame_count = 0

#     while frame_count < sequence_length:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Optional: convert BGR → RGB
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         resized = cv2.resize(frame, (image_width, image_height))
#         frames_array[frame_count] = resized.astype(np.float32) / 255.0
#         frame_count += 1

#     cap.release()

#     # If video is shorter than sequence_length, repeat last frame
#     if frame_count > 0:
#         while frame_count < sequence_length:
#             frames_array[frame_count] = frames_array[frame_count - 1]
#             frame_count += 1

#     return frames_array
