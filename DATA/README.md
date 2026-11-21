# Data Set Establishment Details
This folder contains the data for our project of identifying rock, paper, scissors gestures then creation of a rock, paper, scissors game.

| File | Description |
|---|---|
| `rps_data` | contains 3 sub-folders (one fore each gesture) that contain the images used during our model training, also contains 3 metadata files about each image per gesture stored in a csv (ex: `rock` (folder), `rock_meta.csv`). Created by `SCRIPTS/preprocess.py`  |
| `rps_manifest.csv` | csv file containing all information about every single image location and use: path, label, user_id, hagrid information, split (train vs. val vs. test), and brightness/sharpness of image. Created by `SCRIPTS/preprocess.py`. |

## Data Summary
Our project uses a custom subset of the HaGRIDv2 dataset, focusing on 4,500 hand-crop images across three rock-paper-scissors gestures: fist (rock), palm (paper), and peace (scissors). The preprocessing script (`preprocess.py`) combines the initial gesture metadata, validates every image (checking for readability, 224x224 RGB formatting, and duplicates), and splits the data by user ID so that no participant appears in more than one split (training, validation, or test). The final outputs are stored as `DATA/rps_manifest.csv` and `OUTPUT/split_summary.csv`.

## Provenance
HaGRIDv2 was created by the researchers who developed the HaGRID hand gesture recognition dataset. The images were collected from different individuals to create a dataset that could be used to train and test hand gesture recognition models. Each image was labeled with information about the gesture being performed, along with other details such as hand locations and landmarks on the hand. We obtained the dataset from its public GitHub repository and used the provided annotations rather than making our own labels which would have been too time-consuming. We also inspected the annotated ZIP file through python to understand how the files were organized and identify the available gesture categories. We will select images from the three categories relevant to our project and divide them into training, validation, and test splits ourselves.

## License
The HaGRIDv2 dataset is distributed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license, can check HaGRIDv2 repo for full information. This means that the dataset can be used, shared, and modified as long as the original creators are credited. You are free to share and adapt the data from this project as long as proper credit is given to the original HaGRIDv2 creators.

## Ethical Statements
Since the dataset contains pictures of real people, privacy and responsible data use are important considerations. The dataset includes images of individuals as well as meta information about the person, like demographic information, such as age, gender, and race. Our project focuses on recognizing hand gestures rather than identifying people or studying their personal characteristics. We will not use demographic information to make predictions about individuals. We also are following the dataset's licensing requirements and give proper credit to its creators.

## Data Dictionary

| Feature | Description |
| :--- | :--- |
| `path` | Relative file path to the image within the project folder. |
| `label` | Gesture label (`rock`, `paper`, `scissors`). |
| `split` | Assigned dataset split (`train`, `val`, `test`). |
| `user_id` | Unique identifier for the person in the image (used to prevent data leakage across splits). |
| `hagrid_gesture` | Original HaGRID gesture name (`fist`, `palm`, `peace`). |
| `hagrid_image_id` | Original ID of the source HaGRID image. |
| `hand_index` | Index indicating the specific hand in the source image. |
| `mean_brightness` | Quality measure tracking average image brightness (for analysis only). |
| `sharpness` | Quality measure tracking image sharpness via Laplacian variance (for analysis only). |

## Explanatory Plots

### Plot 1: Image Brightness Distribution by Gesture
![Image Brightness Distribution by Gesture](image)

### Plot 2: Hand Bounding Box Area by Gesture
![Hand Bounding Box Area by Gesture](image2)
