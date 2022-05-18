"""
Firstly, let's import the packages that will be used for this script! 
"""
#operating systems
import os, sys

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt
    

def load_and_process_data():
    """
This function loads and normalizes the cifar10 data, initializes the labelbinarizer and binarizes labels and splits the data into train/test data 
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    X_train=X_train/255
    X_test = X_test/255
    
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    labels = ["airplane", "automobile",
              "bird", "cat",
              "deer","dog",
              "frog","horse",
              "ship","truck"]
    return (X_train, y_train), (X_test, y_test), labels

def define_model():
    """
This function defines the model I will be using for transfer learning (VGG16), turns of the trainable layers in order to use transfer learning and adds the new classifier layers using the softmax activiationlayer, since we're working with multiclass in the CIFAR10 data. Lastly we define that we want to use a categorical loss function, again for the multiclass data, and that we want to optimize the model for accuracy. We use the schedule part to define that we want to take "big steps" at first, and then slow down as the model is training. We're using the stochastic gradient descent algorithm for the model.        
    """
    model = VGG16(include_top = False,
             pooling = "avg",
             input_shape = (32,32,3))
    
    for layer in model.layers:
        layer.trainable = False
        
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(10, activation='softmax')(class1)
     
    model = Model(inputs=model.inputs,
                  outputs=output)
        
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd =SGD(learning_rate=lr_schedule)
    model.compile(optimizer=sgd,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model 

def evaluate(model, X_test, y_test, labels):
    """
This function takes the previously defined model, makes predictions on the test data and saves a classification report to the "out" folder. 
    """
    predictions = model.predict(X_test, batch_size = 128)
    report = classification_report(y_test.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=labels)
    file_path = 'out/classification_report.txt'
    sys.stdout = open(file_path, "w")
    print(report)
    sys.stdout.close()

def plot_history(H, epochs):
    """
This function plots the history of the model, visualizing the loss and accuracy for the test and train data, and saves the plot to the "out" path. 
    """
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    plt.savefig(os.path.join("out", "history.png"))
    
    
    
def main():
    """
The main function defines which functions to run, when the script is run from the terminal. Further this main function includes the training of the model on the training data, and the evaluation of the model.  
    """
    (X_train, y_train), (X_test, y_test), labels = load_and_process_data()
    model = define_model()
    
    H = model.fit(X_train, y_train,
                  validation_data = (X_test, y_test),
                  batch_size = 128,
                  epochs = 10,
                  verbose = 1) 
    
    evaluate(model, X_test, y_test, labels)
    
    plot_history(H,10)
    
    
if __name__== "__main__":
    main()
