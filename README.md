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
1. Clone the repo and download requirements
   - Clone the repo to your local and open it up in your choice of IDE
   - Download the requirements listed in the `requirements.txt` file that is located in the top level directory
**NOTE**: All scripts are run from the top-level directory aka root directory aka the directory that contains `DATA`, `OUTPUT`, `SCRIPTS` folders
2. Run the `eda.py script`, this generates the `annotations.zip` file and `ann` folder.
    - run: python3 SCRIPTS/eda.py
3. Next run `eda_plot.py` to generate the eda plots
    - run: python3 SCRIPTS/eda_plots.py
4. Run preprocess.py to download the image data and process it accordingly
   - run: python3 SCRIPTS/preprocess.py
   - This downloads all of the image data and then it separates it into the train, test, validation split, as well as gets all of the metadata
5. Now, run model.py
   - run: python3 SCRIPTS/model.py
   - This creates and trains the mobileNETv2 model, makes the confusion matrix, and evaluates the model classification results based on different number of layers that are retrained
   - Expected results: the terminal should show
     - Images - train: 3177, val: 694, test: 629
     - a validation table where fine-tuning the top 50 layers has the lowest validation loss (about 0.015), followed by "Keeping the top 50 layers"
     - test accuracy of about 0.994, with recall of about 0.996 for rock, 0.984 for paper, and 1.000 for scissors, and True for both the 80% accuracy and 70% recall goals
     - a majority-class baseline of about 0.356 (always predicting rock)
   - Files saved to OUTPUT: rps_mobilenetv2.keras, confusion_matrix.png, classification_report.csv, fine_tune_search.csv
   - Numbers may differ slightly between computers, but test accuracy should stay well above 80%
6. Finally, run the game
   - run: python3 SCRIPTS/rps_game.py
   - This loads the game and you can play, have fun! Make sure that your gesture fits inside the outlined box, for example if you hand is doing paper but you just have your palm in the box and fingers outside of it that will mess with the classification

## References
[1] A. Nuzhdin, A. Nagaev, A. Sautin, A. Kapitanov, and K. Kvanchiani, “HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition,” arXiv:2412.01508, 2024. [Online]. Available: https://www.alphaxiv.org/abs/2412.01508
[2] A. Kapitanov, K. Kvanchiani, A. Nagaev, R. Kraynov, and A. Makhliarchuk, “HaGRID – HAnd Gesture Recognition Image Dataset,” GitHub repository, 2022. [Online]. Available: https://github.com/hukenovs/hagrid. Accessed: Nov. 6, 2026.
