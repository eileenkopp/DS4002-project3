"""
Train/fine tune and evaluate the rock/paper/scissors MobileNetV2 classifier.
*****MUST BE RUN FROM THE PROJECT ROOT DIRECTORY aka the folder that contains 'DATA/', 'SCRIPTS/', 'OUTPUT/' folders**********

Outputs: 
OUTPUT/rps_mobilenetv2.keras (model for game)
OUTPUT/confusion_matrix.png
OUTPUT/classification_report.csv
OUTPUT/fine_tune_search.csv
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix)
import numpy as np


#Needs to be run from the projects top-level folder aka project root directory, cannot be run from the scripts folder
PROJECT_ROOT = Path(".")

DATA_DIR = PROJECT_ROOT / "DATA"
OUTPUT_DIR = PROJECT_ROOT / "OUTPUT"
MANIFEST_PATH = DATA_DIR / "rps_manifest.csv"
# Image paths inside the manifest look like DATA/rps_data/<label>/<image_id>.jpg and are relative to PROJECT_ROOT.

LABELS = ["rock", "paper", "scissors"]
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# accuracy and recall are fractions between 0 and 1, so the goals are too
GOAL_ACCURACY = 0.80
GOAL_RECALL = 0.70
####################################### loading images/data splits #######################################
def load_images(paths):
    #read all images and returns then as one (N, 224, 224, 3) array (format that model expects)
    images = np.zeros((len(paths), IMG_SIZE, IMG_SIZE, 3), dtype="uint8")  # filled in place to keep memory low
    for i, path in enumerate(paths):
        image = tf.keras.utils.load_img(path)          # opens the .jpg
        images[i] = tf.keras.utils.img_to_array(image)
    return images


def load_split(manifest, split):
    # using the csv manifest, returns the rows of data that match the split and their file paths
    rows = manifest[manifest["split"] == split]
    paths = [str(PROJECT_ROOT / p) for p in rows["path"]]
    labels = rows["label"].map(LABELS.index).to_numpy()
    return paths, labels


####################################### modeling #######################################
def build_model():
    """MobileNetV2 + augmentation + scaling + new 3-class output layer."""

    #imports in the model, don't include the top layer since we are training it, don't train original weights
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    #augmentation for the images to ensure that model works when given non-ideal images
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
    ])


    #takes the images and fits them into the expected imageNet format
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)  #rescales images to imageNet format
    x = base(x, training=False) #runs pretrained model to extract features
    x = tf.keras.layers.GlobalAveragePooling2D()(x) #averages the feature maps with pooling
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x) #sets output to be a softmax to decide between 3 choices
    return tf.keras.Model(inputs, outputs), base #returns the model results


def train(model, x_train, y_train, x_val, y_val, learning_rate, max_epochs):
    model_patience = 3 #used to ensure that model doesn't stop as soon as validation loss doesn't improve
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=model_patience, restore_best_weights=True)
    model.fit(
        x_train, 
        y_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        validation_data=(x_val, y_val), 
        epochs=max_epochs, 
        callbacks=[stop]
    )


####################################### evaluation #######################################
def time_per_image(model, x_test, n=100):
    n = min(n, len(x_test)) # don't run past the end of a small test set
    # warm-up
    for i in range(min(5, n)): 
        model(x_test[i:i + 1], training=False)
    
    # gets the average time to classify one image (ms)
    start = time.perf_counter()
    for i in range(n):
        model(x_test[i:i + 1], training=False)
    return (time.perf_counter() - start) / n * 1000



def evaluate_model(model, x_test, test_labels):
    predicted = model.predict(x_test, batch_size=BATCH_SIZE).argmax(axis=1) #model predictions

    #model prediction accuracy report
    report = classification_report(test_labels, predicted, target_names=LABELS, output_dict=True)
    print("\n" + classification_report(test_labels, predicted, target_names=LABELS))
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "classification_report.csv")

    # majority baseline and time per image, time per image to check so that when it is used in a game it loads quickly
    majority = pd.Series(test_labels).value_counts(normalize=True)
    print(f"Majority-class baseline: {majority.max():.3f} (always predict {LABELS[majority.idxmax()]})")
    print(f"Time per image: {time_per_image(model, x_test):.1f} ms")

    #accuracy and recall information
    accuracy = report["accuracy"]
    recalls = {label: report[label]["recall"] for label in LABELS}
    meets_accuracy = accuracy >= GOAL_ACCURACY
    meets_recall = True
    for r in recalls.values():
        if r < GOAL_RECALL:
            meets_recall = False

    print(f"\nTest accuracy: {accuracy:.3f}")
    print(f"Accuracy >= {GOAL_ACCURACY:.0%}: {meets_accuracy}")
    print(f"\nRecall accuracy: " + ", ".join(f"{k}: {v:.3f}" for k, v in recalls.items()))
    print(f"All Labels recall >= {GOAL_RECALL:.0%}: {meets_recall}")
    return predicted  # reused by plot_confusion_matrix so the model doesn't predict twice




def plot_confusion_matrix(test_labels, predicted, best_num_fine_tune_layers):
    #confusion matrix plot
    matrix = confusion_matrix(test_labels, predicted)
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(cmap="Blues")
    plt.title(f"Model confusion matrix (test-set, {best_num_fine_tune_layers} fine tune layers)")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


####################################### main #######################################
def main():
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)

    #get image paths by split, num images should add up to 4500
    train_paths, train_labels = load_split(manifest, "train")
    val_paths, val_labels = load_split(manifest, "val")
    test_paths, test_labels = load_split(manifest, "test")
    print(f"Images - train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}")


    #gets the images for the splits
    x_train = load_images(train_paths)
    x_val = load_images(val_paths)
    x_test = load_images(test_paths)
    y_train = tf.keras.utils.to_categorical(train_labels, len(LABELS))
    y_val = tf.keras.utils.to_categorical(val_labels, len(LABELS))



    model, base = build_model()
    print("\nStage 1: train the new output layer of the model")
    stage1_lr = 1e-3
    stage1_num_epochs = 10
    train(model, x_train, y_train, x_val, y_val, stage1_lr, stage1_num_epochs)

    stage1_weights = model.get_weights()

    print("\nStage 2: fine-tune the top MobileNetV2 layers")
    stage2_lr = 1e-5   # a lot lower than stage 1 so fine-tuning doesn't wipe out the pretrained features
    stage2_num_epochs = 20
    num_fineTune_layers = [10, 30, 50]
    results, best = [], None

    for n in num_fineTune_layers:
        print(f"Stage 2: fine-tune the top {n} MobileNetV2 layers #######################################")
        model.set_weights(stage1_weights)
        base.trainable = True
        
        # set every layer explicitly: only the top n are trainable, all others frozen.
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= len(base.layers) - n
        train(model, x_train, y_train, x_val, y_val, stage2_lr, stage2_num_epochs)
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

        #save results of number of fine tune layers and compare to see which is best
        results.append({"fine_tune_layers": n, "val_loss": val_loss, "val_acc": val_acc})
        if best is None or val_loss < best["val_loss"]:
            best = {"n": n, "val_loss": val_loss, "weights": model.get_weights()}
        
    
    #save the different fine tune layer results into one csv
    table = pd.DataFrame(results)
    print("\nValidation results by fine-tune depth:")
    print(table.to_string(index=False))
    table.to_csv(OUTPUT_DIR / "fine_tune_search.csv", index=False)


    #save the weights of the best one
    model.set_weights(best["weights"])
    print(f"\nKeeping the top {best['n']} layers (lowest validation loss)")
    model.save(OUTPUT_DIR / "rps_mobilenetv2.keras")

    # only use test set for evaluating the model
    predicted = evaluate_model(model, x_test, test_labels)
    plot_confusion_matrix(test_labels, predicted, best["n"])


if __name__ == "__main__":
    main()