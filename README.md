# Visual_assignment_3
## This is the repository for my third assignment in my Visual Analytics portfolio

### Description of project
This script performs Transfer Learning on the CIFAR-10 dataset. This means that I am using a pretrained convolutional neural network (VGG16) for feature extraction on my image data. Instead of normalizing by flattening and turning the images into black and white, this process turns the images into dense vector representations, thereby retaining more of the original image.The goal of the project is to use this transfer learning for a classification task on the CIFAR-10 dataset. 

### Method 
As mentioned this way of performing classification tasks gets rid of the way we usually normalize - this means that all I've done with the data in this script is load it, split it and normalize it by dividing it by 255. After doing this I just binarize my labels, and then the data is ready for the pretrained VGG16 model to perform the transfer learning. 
I then use keras from tensorflow to define the model and that I don't want the to train the layers, as this would defeat the purpose of the transfer learning. After having defined and saved my new model, I use Scikit-learn to make a classification report, which is then saved to my "out" folder. I also save a figure of the plotted accuracy and loss of the model on the training vs testing data.

### Usage
In order to reproduce my results, a few steps has to be followed:

1) Install the relevant packages - the list of the prerequisites for the script can be found in the requirements.txt
2) Make sure to place the script in the "src" folder. The data used in my code (cifar10) is fetched from tensorflow using the load_data() so nothing has to go into the in-folder. Had you wanted to use the script on a similar dataset, you would have to change up the loading of the data part of the script and place the data in the "in" folder. For this script using the cifar10 dataset, however, the "in" folder is redundant.
3) Run the script from the terminal. Make sure that you are in the main folder when excecuting by typing in "python src/transfer_learning.py"

This should give you approximately the same results as I have gotten in the "out" folder. 


### Results
The results of this script is actually not as good as I would have first expected, considering my previous results when I did the same task using CNN’s, and where I achieved a maximum score of 57%. For the transfer learning I gain a maximum precision score around 60% and a minimum around 40% this doesn't seem like the perfect classification tool, although I would have expected a bigger increase in accuracy when comparing the two methods. 
This could be a matter of redefining the parameters (e.g. learning rate, batch size, etc.), in order to optimize the model. This is something that could be fiddled around with, in order to find the most optimal parameters for the model performance. Also, I am only running it for 10 epochs - if this number was extended the results could also potentially increase in accuracy. Bearing in mind that we are classifying on ten labels though, the accuracy is pretty good.
Looking at the saved plots of the accuracy and loss for the train and test data compared, this looks pretty solid. This can be read from the plots by looking at how close the plotted curve for the train- and test data is to each other - for my plots they are almost identical, which means that the model doesn't seem to be neither over- nor underfitting, which is great! 




