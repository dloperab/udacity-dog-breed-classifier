# Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Write your Algorithm
* [Step 6](#step6): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

Make sure that you've downloaded the required human and dog datasets:
* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dogImages`. 

* Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home directory, at location `/lfw`.  

*Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.*

In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays `human_files` and `dog_files`.


```python
import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("lfw/*/*"))
dog_files = np.array(glob("dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
```

    There are 13233 total human images.
    There are 8351 total dog images.


<a id='step1'></a>
## Step 1: Detect Humans

In this section, we use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  

OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.  In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](dog_app_files/dog_app_3_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 
(You can print out your results and/or write your percentages in this cell)
- 100 faces in Human images = 100%
- 9 faces in Dogs images = 9%


```python
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
humans_counter = len([x for x in tqdm(range(len(human_files_short))) if face_detector(human_files_short[x])])
dogs_counter = len([x for x in tqdm(range(len(dog_files_short))) if face_detector(dog_files_short[x])])

print(f"Faces in Human Images = {(humans_counter/len(human_files_short))*100}%")
print(f"Faces in Dogs Images = {(dogs_counter/len(dog_files_short))*100}%")
```

    100%|██████████| 100/100 [00:01<00:00, 77.77it/s]
    100%|██████████| 100/100 [00:05<00:00, 18.20it/s]

    Faces in Human Images = 100.0%
    Faces in Dogs Images = 9.0%


    


We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.


```python
### (Optional) 
### TODO: Test performance of another face detection algorithm.
### Feel free to use as many code cells as needed.

# Note: The implementation was based in the blog post from PyImageSearch 
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
prototxt = "dl_face_detector_model/deploy.prototxt.txt"
model = "dl_face_detector_model/res10_300x300_ssd_iter_140000.caffemodel"
threshold = 0.9

# loading the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

def dl_face_detector(img_path):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image = cv2.imread(img_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    
    faces_found_counter = 0

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > threshold:
            faces_found_counter += 1 

    return faces_found_counter > 0

# measure performance in deep learning face detector with confidence >= 90%
humans_counter = len([x for x in tqdm(range(len(human_files_short))) if dl_face_detector(human_files_short[x])])
dogs_counter = len([x for x in tqdm(range(len(dog_files_short))) if dl_face_detector(dog_files_short[x])])

print(f"Faces in Human Images = {(humans_counter/len(human_files_short))*100}%")
print(f"Faces in Dogs Images = {(dogs_counter/len(dog_files_short))*100}%")
```

    100%|██████████| 100/100 [00:01<00:00, 61.52it/s]
    100%|██████████| 100/100 [00:01<00:00, 50.50it/s]

    Faces in Human Images = 100.0%
    Faces in Dogs Images = 16.0%


    


---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a [pre-trained model](http://pytorch.org/docs/master/torchvision/models.html) to detect dogs in images.  

### Obtain Pre-trained VGG-16 Model

The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  


```python
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()
print(f"CUDA available: {use_cuda}")

device = 'cuda:0' if use_cuda else 'cpu'
print(f"Device: {device}")

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
```

    CUDA available: True
    Device: cuda:0


Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

### (IMPLEMENTATION) Making Predictions with a Pre-trained Model

In the next code cell, you will write a function that accepts a path to an image (such as `'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.

Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).


```python
from PIL import Image
import torchvision.transforms as transforms

# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    # Image size required for pre-trained models (224 px)
    image_size = 224
    
    # Load the image
    image = Image.open(img_path)
    # Define transforms for the image
    transform = transforms.Compose([transforms.Resize(image_size), 
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # Apply transforms to image
    image = transform(image)
    # Flatten image, load to cuda (if available) and predict
    image = image.unsqueeze(0)
    if use_cuda:
        image = image.to("cuda")
    
    # Predict
    VGG16.eval()
    output = VGG16(image)
    if use_cuda:
        output = output.to("cpu")
    # Get predicted index class
    predicted_index = output.data.numpy().argmax()    
    
    return predicted_index
```

### (IMPLEMENTATION) Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).

Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    output = VGG16_predict(img_path)
    
    return (output >= 151 and output <= 268)
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 2:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 
- Dogs detected in Human Images = 1.0%
- Dogs detected in Dogs Images = 99.0%


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
humans_counter = len([x for x in tqdm(range(len(human_files_short))) if dog_detector(human_files_short[x])])
dogs_counter = len([x for x in tqdm(range(len(dog_files_short))) if dog_detector(dog_files_short[x])])

print(f"(VGG-16) Dogs detected in Human Images = {(humans_counter/len(human_files_short))*100}%")
print(f"(VGG-16) Dogs detected in Dogs Images = {(dogs_counter/len(dog_files_short))*100}%")
```

    100%|██████████| 100/100 [00:01<00:00, 60.70it/s]
    100%|██████████| 100/100 [00:02<00:00, 48.50it/s]

    (VGG-16) Dogs detected in Human Images = 1.0%
    (VGG-16) Dogs detected in Dogs Images = 99.0%


    


We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](http://pytorch.org/docs/master/torchvision/models.html#inception-v3), [ResNet-50](http://pytorch.org/docs/master/torchvision/models.html#id3), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.


```python
### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.

# Get Inception-V3 pre-trained model
inception = models.inception_v3(pretrained=True)

# Move model to GPU if CUDA is available
if use_cuda:
    inception = inception.cuda()
    
def inception_v3_dog_detector(img_path):
    '''
    Use pre-trained Inception-V3 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        returns "True/False" if a dog is detected in the image stored at img_path
    '''
    # Image size required for pre-trained models (224 px)
    image_size = 224
    
    # Load the image
    image = Image.open(img_path)
    # Define transforms for the image
    transform = transforms.Compose([transforms.Resize(image_size), 
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # Apply transforms to image
    image = transform(image)
    # Flatten image, load to cuda (if available) and predict
    image = image.unsqueeze(0)
    if use_cuda:
        image = image.to("cuda")
    
    # Predict
    inception.eval()
    output = inception(image)
    if use_cuda:
        output = output.to("cpu")
    # Get predicted index class
    predicted_index = output.data.numpy().argmax()    
    
    return (predicted_index >= 151 and predicted_index <= 268)
```


```python
# Test Inception-V3 Dog Detector
humans_counter = len([x for x in tqdm(range(len(human_files_short))) if inception_v3_dog_detector(human_files_short[x])])
dogs_counter = len([x for x in tqdm(range(len(dog_files_short))) if inception_v3_dog_detector(dog_files_short[x])])
        
print(f"(Inception-V3) Dogs detected in Human Images = {(humans_counter/len(human_files_short))*100}%")
print(f"(Inception-V3) Dogs detected in Dogs Images = {(dogs_counter/len(dog_files_short))*100}%")
```

    100%|██████████| 100/100 [00:01<00:00, 57.89it/s]
    100%|██████████| 100/100 [00:02<00:00, 44.68it/s]

    (Inception-V3) Dogs detected in Human Images = 3.0%
    (Inception-V3) Dogs detected in Dogs Images = 99.0%


    


---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 10%.  In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively).  You may find [this documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!


```python
import os
from torchvision import datasets

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

# Number of subprocesses to use for data loading
num_workers = 0
# Number of samples per batch to load
batch_size = 20
# Image size
image_size = (224, 224)

transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(30),                                      
                                      transforms.Resize(image_size),                                      
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

transform_valid = transforms.Compose([transforms.Resize(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder('dogImages/train',transform=transform_train)
val_data = datasets.ImageFolder('dogImages/valid',transform=transform_valid)
test_data = datasets.ImageFolder('dogImages/test',transform=transform_valid)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

loaders_scratch = {'train': train_loader,
                   'valid': val_loader,
                   'test': test_loader}

classes = train_data.classes
```

**Question 3:** Describe your chosen procedure for preprocessing the data. 
- How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
- Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?


**Answer**:
- The images are resize and normalize using standard normalization. 
- The size that I picked for the input tensor is 224x224 px, because smaller sizes helps to minimize the processing time in the training loop.
- I've used data augmentation for training data. The augmentations made were RandomHorizontalFlip and RandomRotation. Data augmentation is useful to prevent overfitting.

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  Use the template in the code cell below.


```python
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        # 5 Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        
        # batch norm layers
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5_bn = nn.BatchNorm2d(512)        
        self.fc1_bn = nn.BatchNorm1d(1024)
    
        # max pooling layer
        self.max_pool = nn.MaxPool2d(2, 2)
        
        # linear layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.output = nn.Linear(1024, len(classes))
        
        # dropout layer
        self.dropout_convs = nn.Dropout(0.2)
        self.dropout_linear = nn.Dropout(0.4)
    
    def forward(self, x):
        ## Define forward behavior
        
        # 1st conv layer set
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.max_pool(x)
        x = self.dropout_convs(x)
        # 2nd conv layer set
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.max_pool(x)
        x = self.dropout_convs(x)       
        # 3rd conv layer set
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.max_pool(x)
        x = self.dropout_convs(x)
        # 4th conv layer set
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = self.max_pool(x)
        x = self.dropout_convs(x)
        # 5th conv layer set
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = self.max_pool(x)
        x = self.dropout_convs(x)
        
        # flatten image input
        x = x.view(-1, 512 * 7 * 7)
        
        # 1st hidden layer, with relu activation function and dropout
        x = F.relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.dropout_linear(x)
        
        # output layer
        x = self.output(x)
        
        return x

#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()

print(model_scratch)
```

    Net(
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv5): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv1_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3_bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv4_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv5_bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc1_bn): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (max_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=25088, out_features=1024, bias=True)
      (output): Linear(in_features=1024, out_features=133, bias=True)
      (dropout_convs): Dropout(p=0.2, inplace=False)
      (dropout_linear): Dropout(p=0.4, inplace=False)
    )


__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

__Answer:__ 

- I implemented 5 convolutional layers to detect patterns in the images. The combination that I choose was INPUT => CONV => BATCH NORM => RELU. The kernel size of 3x3 and max pooling layer of 2x2 for reduce the matrix size.
- The final convolutional layer ended with a layer of 512 * 7 * 7.
- I implemented 1 hidden layer (linear layer) adding dropout (40%) to prevent overfitting and applying ReLU activation function.
- I implemented an output layer to produce the classification for the image to the 133 possible classes (dog breeds).

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and the optimizer as `optimizer_scratch` below.


```python
import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.01)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.


```python
# the following import is required for training to be robust to truncated images
import numpy as np

from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # calculate the train loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            
            # forward pass to predict the outputs
            output = model(data)
            # compute the loss
            loss = criterion(output,target)
            # update the average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # print training/validation statistics 
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}")
        
        ## save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model

# train the model
model_scratch = train(70, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
```

    Epoch: 1 	Training Loss: 4.708668 	Validation Loss: 5.466578
    Validation loss decreased (inf --> 5.466578).  Saving model ...
    Epoch: 2 	Training Loss: 4.291674 	Validation Loss: 5.871588
    Epoch: 3 	Training Loss: 4.073150 	Validation Loss: 6.101068
    Epoch: 4 	Training Loss: 3.899866 	Validation Loss: 5.573841
    Epoch: 5 	Training Loss: 3.792346 	Validation Loss: 5.338641
    Validation loss decreased (5.466578 --> 5.338641).  Saving model ...
    Epoch: 6 	Training Loss: 3.671974 	Validation Loss: 5.014486
    Validation loss decreased (5.338641 --> 5.014486).  Saving model ...
    Epoch: 7 	Training Loss: 3.583745 	Validation Loss: 4.951972
    Validation loss decreased (5.014486 --> 4.951972).  Saving model ...
    Epoch: 8 	Training Loss: 3.479955 	Validation Loss: 4.679465
    Validation loss decreased (4.951972 --> 4.679465).  Saving model ...
    Epoch: 9 	Training Loss: 3.372345 	Validation Loss: 4.612576
    Validation loss decreased (4.679465 --> 4.612576).  Saving model ...
    Epoch: 10 	Training Loss: 3.276541 	Validation Loss: 4.551418
    Validation loss decreased (4.612576 --> 4.551418).  Saving model ...
    Epoch: 11 	Training Loss: 3.187778 	Validation Loss: 4.636153
    Epoch: 12 	Training Loss: 3.124188 	Validation Loss: 3.941701
    Validation loss decreased (4.551418 --> 3.941701).  Saving model ...
    Epoch: 13 	Training Loss: 3.027850 	Validation Loss: 188.423767
    Epoch: 14 	Training Loss: 2.951499 	Validation Loss: 3.895864
    Validation loss decreased (3.941701 --> 3.895864).  Saving model ...
    Epoch: 15 	Training Loss: 2.860928 	Validation Loss: 3.983988
    Epoch: 16 	Training Loss: 2.773514 	Validation Loss: 3.922044
    Epoch: 17 	Training Loss: 2.742068 	Validation Loss: 3.692452
    Validation loss decreased (3.895864 --> 3.692452).  Saving model ...
    Epoch: 18 	Training Loss: 2.669441 	Validation Loss: 3.553144
    Validation loss decreased (3.692452 --> 3.553144).  Saving model ...
    Epoch: 19 	Training Loss: 2.577922 	Validation Loss: 3.712995
    Epoch: 20 	Training Loss: 2.537939 	Validation Loss: 3.549377
    Validation loss decreased (3.553144 --> 3.549377).  Saving model ...
    Epoch: 21 	Training Loss: 2.471517 	Validation Loss: 3.762792
    Epoch: 22 	Training Loss: 2.391424 	Validation Loss: 3.382495
    Validation loss decreased (3.549377 --> 3.382495).  Saving model ...
    Epoch: 23 	Training Loss: 2.335025 	Validation Loss: 3.242957
    Validation loss decreased (3.382495 --> 3.242957).  Saving model ...
    Epoch: 24 	Training Loss: 2.266449 	Validation Loss: 3.295328
    Epoch: 25 	Training Loss: 2.214181 	Validation Loss: 3.597195
    Epoch: 26 	Training Loss: 2.167599 	Validation Loss: 3.324264
    Epoch: 27 	Training Loss: 2.099338 	Validation Loss: 3.113283
    Validation loss decreased (3.242957 --> 3.113283).  Saving model ...
    Epoch: 28 	Training Loss: 2.072892 	Validation Loss: 3.571359
    Epoch: 29 	Training Loss: 2.018410 	Validation Loss: 3.197495
    Epoch: 30 	Training Loss: 1.952151 	Validation Loss: 3.328104
    Epoch: 31 	Training Loss: 1.923804 	Validation Loss: 3.162748
    Epoch: 32 	Training Loss: 1.862601 	Validation Loss: 3.360185
    Epoch: 33 	Training Loss: 1.824155 	Validation Loss: 3.290107
    Epoch: 34 	Training Loss: 1.817311 	Validation Loss: 3.083587
    Validation loss decreased (3.113283 --> 3.083587).  Saving model ...
    Epoch: 35 	Training Loss: 1.747753 	Validation Loss: 3.232751
    Epoch: 36 	Training Loss: 1.721183 	Validation Loss: 3.115296
    Epoch: 37 	Training Loss: 1.648835 	Validation Loss: 3.206422
    Epoch: 38 	Training Loss: 1.632086 	Validation Loss: 3.170784
    Epoch: 39 	Training Loss: 1.617722 	Validation Loss: 2.998379
    Validation loss decreased (3.083587 --> 2.998379).  Saving model ...
    Epoch: 40 	Training Loss: 1.563896 	Validation Loss: 3.097397
    Epoch: 41 	Training Loss: 1.554640 	Validation Loss: 3.033855
    Epoch: 42 	Training Loss: 1.498507 	Validation Loss: 3.168643
    Epoch: 43 	Training Loss: 1.497337 	Validation Loss: 3.031794
    Epoch: 44 	Training Loss: 1.434181 	Validation Loss: 2.926357
    Validation loss decreased (2.998379 --> 2.926357).  Saving model ...
    Epoch: 45 	Training Loss: 1.391057 	Validation Loss: 3.070295
    Epoch: 46 	Training Loss: 1.329908 	Validation Loss: 3.021247
    Epoch: 47 	Training Loss: 1.299828 	Validation Loss: 2.898281
    Validation loss decreased (2.926357 --> 2.898281).  Saving model ...
    Epoch: 48 	Training Loss: 1.262918 	Validation Loss: 2.992808
    Epoch: 49 	Training Loss: 1.256465 	Validation Loss: 2.982302
    Epoch: 50 	Training Loss: 1.240758 	Validation Loss: 3.033550
    Epoch: 51 	Training Loss: 1.205828 	Validation Loss: 2.877741
    Validation loss decreased (2.898281 --> 2.877741).  Saving model ...
    Epoch: 52 	Training Loss: 1.167989 	Validation Loss: 2.909569
    Epoch: 53 	Training Loss: 1.100165 	Validation Loss: 3.154155
    Epoch: 54 	Training Loss: 1.119028 	Validation Loss: 2.884517
    Epoch: 55 	Training Loss: 1.092821 	Validation Loss: 2.917532
    Epoch: 56 	Training Loss: 1.042678 	Validation Loss: 3.061168
    Epoch: 57 	Training Loss: 1.055609 	Validation Loss: 3.219753
    Epoch: 58 	Training Loss: 1.006845 	Validation Loss: 2.922692
    Epoch: 59 	Training Loss: 0.978064 	Validation Loss: 2.964783
    Epoch: 60 	Training Loss: 0.985855 	Validation Loss: 2.911660
    Epoch: 61 	Training Loss: 0.955508 	Validation Loss: 2.906320
    Epoch: 62 	Training Loss: 0.942283 	Validation Loss: 2.879503
    Epoch: 63 	Training Loss: 0.904760 	Validation Loss: 2.834675
    Validation loss decreased (2.877741 --> 2.834675).  Saving model ...
    Epoch: 64 	Training Loss: 0.897020 	Validation Loss: 2.844318
    Epoch: 65 	Training Loss: 0.873605 	Validation Loss: 2.851314
    Epoch: 66 	Training Loss: 0.850129 	Validation Loss: 2.886626
    Epoch: 67 	Training Loss: 0.827643 	Validation Loss: 2.897169
    Epoch: 68 	Training Loss: 0.806804 	Validation Loss: 3.172218
    Epoch: 69 	Training Loss: 0.793364 	Validation Loss: 2.994290
    Epoch: 70 	Training Loss: 0.779002 	Validation Loss: 2.984097





    <All keys matched successfully>



### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.


```python
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
```

    Test Loss: 2.692244
    
    
    Test Accuracy: 34% (290/836)


---
<a id='step4'></a>
## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively). 

If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.


```python
## TODO: Specify data loaders

# Number of subprocesses to use for data loading
num_workers = 0
# Number of samples per batch to load
batch_size = 20
# Image size
image_size = (224, 224)

transform_train_tl = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(30),   
                                         transforms.Resize(image_size),  
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225])])

transform_valid_tl = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225])])

train_data_tl = datasets.ImageFolder('dogImages/train',transform=transform_train_tl)
val_data_tl = datasets.ImageFolder('dogImages/valid',transform=transform_valid_tl)
test_data_tl = datasets.ImageFolder('dogImages/test',transform=transform_valid_tl)

train_loader_tl = torch.utils.data.DataLoader(train_data_tl, batch_size=batch_size, shuffle = True)
val_loader_tl = torch.utils.data.DataLoader(val_data_tl, batch_size=batch_size)
test_loader_tl = torch.utils.data.DataLoader(test_data_tl, batch_size=batch_size)

loaders_transfer = {'train': train_loader_tl,
                    'valid': val_loader_tl,
                    'test': test_loader_tl}

classes = train_data_tl.classes
```

### (IMPLEMENTATION) Model Architecture

Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.


```python
import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 

# define restnet50 model
model_transfer = models.resnet50(pretrained=True)
# freeze all the network except the final layer
for param in model_transfer.parameters():
    param.requires_grad = False

# get inputs for final layer and set the output to the classes needed
inputs = model_transfer.fc.in_features
model_transfer.fc = nn.Linear(inputs, len(classes))

# check if CUDA is available
use_cuda = torch.cuda.is_available()
print(f"CUDA available: {use_cuda}")

device = 'cuda:0' if use_cuda else 'cpu'
print(f"Device: {device}")

# move model to GPU if CUDA is available
if use_cuda:
    model_transfer = model_transfer.cuda()
```

    CUDA available: True
    Device: cuda:0


__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 

- I selected the resnet50 architecture in conjunction with SGD optimizer after try several configurations with Adam/SGD optimizers with resnet50 and vgg16.
- I got a good accuracy for this project, but obviously it can be improved. Something to try is weight initializations, optimizers, learnings rates, add momentum, etc.
- I think restnet50 architecture can achieve good result for classification problems. It is strong and the train process isn't as long as others heavy models.

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.


```python
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr=0.01)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.


```python
# train the model
model_transfer = train(40, loaders_transfer, model_transfer, optimizer_transfer, 
                       criterion_transfer, use_cuda, 'model_transfer.pt')
```

    Epoch: 1 	Training Loss: 0.105016 	Validation Loss: 0.498730
    Validation loss decreased (inf --> 0.498730).  Saving model ...
    Epoch: 2 	Training Loss: 0.095514 	Validation Loss: 0.492000
    Validation loss decreased (0.498730 --> 0.492000).  Saving model ...
    Epoch: 3 	Training Loss: 0.093382 	Validation Loss: 0.483567
    Validation loss decreased (0.492000 --> 0.483567).  Saving model ...
    Epoch: 4 	Training Loss: 0.099613 	Validation Loss: 0.485374
    Epoch: 5 	Training Loss: 0.090826 	Validation Loss: 0.465601
    Validation loss decreased (0.483567 --> 0.465601).  Saving model ...
    Epoch: 6 	Training Loss: 0.092723 	Validation Loss: 0.480398
    Epoch: 7 	Training Loss: 0.093269 	Validation Loss: 0.475643
    Epoch: 8 	Training Loss: 0.095010 	Validation Loss: 0.469087
    Epoch: 9 	Training Loss: 0.089322 	Validation Loss: 0.473285
    Epoch: 10 	Training Loss: 0.094585 	Validation Loss: 0.482770
    Epoch: 11 	Training Loss: 0.092906 	Validation Loss: 0.480597
    Epoch: 12 	Training Loss: 0.092493 	Validation Loss: 0.463594
    Validation loss decreased (0.465601 --> 0.463594).  Saving model ...
    Epoch: 13 	Training Loss: 0.090248 	Validation Loss: 0.481624
    Epoch: 14 	Training Loss: 0.090079 	Validation Loss: 0.479980
    Epoch: 15 	Training Loss: 0.089184 	Validation Loss: 0.485307
    Epoch: 16 	Training Loss: 0.092139 	Validation Loss: 0.478807
    Epoch: 17 	Training Loss: 0.087166 	Validation Loss: 0.473806
    Epoch: 18 	Training Loss: 0.095281 	Validation Loss: 0.475179
    Epoch: 19 	Training Loss: 0.098456 	Validation Loss: 0.482924
    Epoch: 20 	Training Loss: 0.088382 	Validation Loss: 0.468759
    Epoch: 21 	Training Loss: 0.088721 	Validation Loss: 0.478288
    Epoch: 22 	Training Loss: 0.094221 	Validation Loss: 0.473615
    Epoch: 23 	Training Loss: 0.093188 	Validation Loss: 0.492883
    Epoch: 24 	Training Loss: 0.098016 	Validation Loss: 0.479410
    Epoch: 25 	Training Loss: 0.089406 	Validation Loss: 0.475916
    Epoch: 26 	Training Loss: 0.087326 	Validation Loss: 0.482681
    Epoch: 27 	Training Loss: 0.097531 	Validation Loss: 0.479601
    Epoch: 28 	Training Loss: 0.091088 	Validation Loss: 0.488183
    Epoch: 29 	Training Loss: 0.097723 	Validation Loss: 0.490436
    Epoch: 30 	Training Loss: 0.093556 	Validation Loss: 0.472185
    Epoch: 31 	Training Loss: 0.084967 	Validation Loss: 0.481130
    Epoch: 32 	Training Loss: 0.092743 	Validation Loss: 0.464473
    Epoch: 33 	Training Loss: 0.094442 	Validation Loss: 0.493410
    Epoch: 34 	Training Loss: 0.090411 	Validation Loss: 0.485025
    Epoch: 35 	Training Loss: 0.089229 	Validation Loss: 0.473460
    Epoch: 36 	Training Loss: 0.090532 	Validation Loss: 0.482454
    Epoch: 37 	Training Loss: 0.091550 	Validation Loss: 0.474270
    Epoch: 38 	Training Loss: 0.090252 	Validation Loss: 0.475301
    Epoch: 39 	Training Loss: 0.086601 	Validation Loss: 0.478467
    Epoch: 40 	Training Loss: 0.082952 	Validation Loss: 0.471941


### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.


```python
# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))

# test the model
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```

    Test Loss: 0.546081
    
    
    Test Accuracy: 85% (713/836)


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in train_data_tl.classes]

def predict_breed_transfer(img_path):
    if model_transfer.train:
        model_transfer.eval()
    
    # load the image and return the predicted breed
    image = Image.open(img_path).convert("RGB")
    image = transform_valid_tl(image).unsqueeze(0)
    if use_cuda:
        image = image.cuda()
    output = model_transfer(image)
    
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())
    
    return class_names[preds]
```

---
<a id='step5'></a>
## Step 5: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 4 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    
    if(dog_detector(img_path)):
        return f"Dog detected! You look like a ... {predict_breed_transfer(img_path)}"
    
    if (face_detector(img_path)):
        return f"Human detected! You could be a ... {predict_breed_transfer(img_path)}"
    
    return "Nothing detected. What are you?"
```

---
<a id='step6'></a>
## Step 6: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that _you_ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ 

- The outputs obtained was really nice. I expected a worse performance since the dataset is complex, but I'm surprised and very happy with the results.
- Possible points of improvements:
 * Try more data augmentations techniques.
 * Explore others pretrained models.
 * Explore hyperparameters like momentum, different learning rates and maybe a learning rate finder. Also try different optimizers.
 * Also I would to fine-tune the model and configure some weights initializations.


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

test_files = np.array(glob("test_images/*"))

index = 1
rows = test_files.size / 2

fig = plt.figure(figsize=(30, 24))
fig.suptitle("Predictor Results", fontsize=20, fontweight="bold")

for file in test_files:
    result = run_app(file)    
    
    ax = fig.add_subplot(rows, 2, index)
    ax.set_title(result, fontsize=16)
    
    img = Image.open(file)
    ax.imshow(img)
    
    index += 1
```


![png](dog_app_files/dog_app_56_0.png)



```python

```
