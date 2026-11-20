# DS4002-project3

## Content Summary
This repository contains our data, scripts, and results/output for our project on image data: photos of hand gestures relabeled as rock, paper, and scissors. We start with the HaGRID hand-gesture dataset [1], [2] and process it into our own dataset by selecting the fist, palm, and peace gestures, relabeling them as rock, paper, and scissors, cropping each hand from its bounding box, and splitting the images by person into training, validation, and test sets. We then fine-tune MobileNetV2, a pretrained convolutional neural network, to see whether it can classify the gestures with at least 80% accuracy on people it was not trained on, and save the trained model, a confusion matrix, and a classification report into the output folder. Finally, we use the model in an interactive webcam game where a player competes against the computer in rock-paper-scissors.

## Section 1: Software and Platform Section

## Section 2: A Map of Your Documentation

## Section 3: Instructions for Reproducing Results

## References
[1] A. Nuzhdin, A. Nagaev, A. Sautin, A. Kapitanov, and K. Kvanchiani, “HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition,” arXiv:2412.01508, 2024. [Online]. Available: https://www.alphaxiv.org/abs/2412.01508
[2] A. Kapitanov, K. Kvanchiani, A. Nagaev, R. Kraynov, and A. Makhliarchuk, “HaGRID – HAnd Gesture Recognition Image Dataset,” GitHub repository, 2022. [Online]. Available: https://github.com/hukenovs/hagrid. Accessed: Nov. 6, 2026.
