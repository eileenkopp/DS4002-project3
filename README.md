# DS4002-project3

## Content Summary
This repository contains our data, scripts, and results/output for our project on image data: photos of hand gestures relabeled as rock, paper, and scissors. We start with the HaGRID hand-gesture dataset [1], [2] and process it into our own dataset by selecting the fist, palm, and peace gestures, relabeling them as rock, paper, and scissors, cropping each hand from its bounding box, and splitting the images by person into training, validation, and test sets. We then fine-tune MobileNetV2, a pretrained convolutional neural network, to see whether it can classify the gestures with at least 80% accuracy on people it was not trained on, and save the trained model, a confusion matrix, and a classification report into the output folder. Finally, we use the model in an interactive webcam game where a player competes against the computer in rock-paper-scissors.

## Section 1: Software and Platform Section
- We used Python for this project, specifically Python 3.14.7

Add-on packages that need to be installed with the software (can check requirements.txt to see exact versions):
- remotezip
- Pillow
- pandas
- numpy
- matplotlib
- scikit-learn
- opencv-python
- random
- tensorflow

The platform we used: 
- Mac

## Section 2: A Map of Your Documentation
- DATA
    - rps_data
      -  rock (folder that contains all training rock images)
      -  paper (folder that contains all training paper images)
      -  scissors (folder that contains all scissor training images)
      -  rock_meta.csv
      -  paper_meta.csv
      -  scissor_meta.csv
    - rps_manifest.csv
    - annotations.zip (**too large to upload**, see `DATA/README.md` for more info)
    - ann (**too large to upload**, see `DATA/README.md` for more info)
      - annotations
        - test (contains a json file of each category)
          - call.json
          - .... (json files by category)
        - train (same as test folder)
          - ... (json files by category)
        - val (same as test and train folder)
          - ... (json files by category)
- OUTPUT
    - brightness_distributions.png
    - relative_hand_area_distributions.png
    - confusion_matrix.png
    - classification_report.csv
    - fine_tune_search.csv
    - rps_mobilenetv2.keras
    - split_summary.csv
- SCRIPTS
    - eda.py
    - eda_plots.py
    - game_logic.py
    - gesture_classifier.py
    - model.py
    - preprocess.py
    - rps_game.py
- LICENSE
- README.md
- requirements.txt

## Section 3: Instructions for Reproducing Results

## References
[1] A. Nuzhdin, A. Nagaev, A. Sautin, A. Kapitanov, and K. Kvanchiani, “HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition,” arXiv:2412.01508, 2024. [Online]. Available: https://www.alphaxiv.org/abs/2412.01508
[2] A. Kapitanov, K. Kvanchiani, A. Nagaev, R. Kraynov, and A. Makhliarchuk, “HaGRID – HAnd Gesture Recognition Image Dataset,” GitHub repository, 2022. [Online]. Available: https://github.com/hukenovs/hagrid. Accessed: Nov. 6, 2026.
