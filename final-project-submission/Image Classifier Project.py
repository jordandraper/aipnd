
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import torch
import json 

import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from collections import OrderedDict

from PIL import Image


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# using dictionary to make calls simpler
dirs = {'train' : train_dir, 'valid' : valid_dir, 'test' : test_dir}


# In[36]:


# TODO: Define your transforms for the training, validation, and testing sets
# transforms obtained from https://pytorch.org/docs/stable/torchvision/transforms.html
data_transforms = {
                'train':transforms.Compose([transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                'valid':transforms.Compose([transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                'test':transforms.Compose([transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
                    }

# TODO: Load the datasets with ImageFolder
# originally had each dataset and dataloader defined individually, then was inspired by these examples 
# https://www.datacamp.com/community/tutorials/python-dictionary-comprehension to consolidate into a dictionary
datasets = {x : datasets.ImageFolder(dirs[x],transform = data_transforms[x]) for x in ['train', 'valid', 'test']}


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x : torch.utils.data.DataLoader(datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}            

# dataset sizes to be used in accuracy computations
dataset_sizes = {x: len(datasets[x]) 
                              for x in ['train', 'valid', 'test']}


# In[37]:


# test to check the above code
dataiter = iter(dataloaders['test'])
images, labels = dataiter.next()
plt.imshow(images[0,0].numpy().squeeze(), cmap='Greys_r');
images.size()


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[38]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.
# 
# Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

# In[6]:


# TODO: Build and train your network
# originally had several hidden layers but decided to reduce to just one

vgg16 = models.vgg16(pretrained=True)
hidden_layers = [4096]

# freezing parameters as advised in nanodegree lectures
for param in vgg16.parameters():
    param.requires_grad = False
    
# creating classfier to be used with model. provided in nanodegree
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_layers[0])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layers[0], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# set classifier
vgg16.classifier = classifier

# I followed this documentation https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
#  to create model. I decided to include a scheduler for the learning rate even though it is not mentioned in previous videos

# set criterion
criterion = nn.NLLLoss()

# set optimizer
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)

# learned more about scheduler here https://pytorch.org/docs/stable/optim.html
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# setting number of epochs
epochs = 10

# much of this code is based on nanodegree labs and documentation 
# from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
def train_model(model, criterion, optimizer, scheduler, epochs=20, device = 'cuda'):

    # change to device for GPU
    model.to(device)

    for e in range(epochs):
        print("Epoch: {}/{}... ".format(e+1, epochs))
              
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0
            running_corrects = 0
            
            for ii, (inputs, labels) in enumerate(dataloaders[phase]):

                # Move input and label tensors to the GPU
                inputs, labels = inputs.to(device), labels.to(device)
                
                # zero parameter gradients
                optimizer.zero_grad()

                # Forward and backward passes
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
        print()
    return model

tr_model = train_model(vgg16, criterion, optimizer, scheduler, 10)


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[39]:


# TODO: Do validation on the test set
# most of this code was provided already in nano degree
def calc_accuracy(model, dataloader, device = 'cuda'):
    correct = 0
    total = 0
    model.eval()
    
    #sending to gpu
    model.to(device)
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            # Move images and label tensors to the GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

calc_accuracy(vgg16,dataloaders['test'])


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[40]:


# TODO: Save the checkpoint
# most of this code was provided already in nano degree
def save_checkpoint(model):
    model.class_to_idx = datasets['train'].class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_dict': optimizer.state_dict(),
                  'hidden_layers': hidden_layers,
                  }

    torch.save(checkpoint, 'checkpoint.pth')

save_checkpoint(vgg16)


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[41]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
# most of this code was provided already in nano degree
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint['hidden_layers'][0], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# testing the function
load_checkpoint('checkpoint.pth')


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[42]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    # test which side is longer and adjust it down to 256 while preserving ratio
    if im.size[0] > im.size[1]:
        im.thumbnail((1000, 256))
    else:
        im.thumbnail((256, 1000))
    
    # this portion was determined with help from https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    width,height = im.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im)/255.0
    np_image = (np_image - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    np_image = np_image.transpose(2,0,1)
    return np_image

test = process_image(data_dir + '/test/28/image_05230.jpg')


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[43]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    # added title if provided
    if title:
        plt.title(title)
        
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

imshow(test)


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[44]:


def predict(image_path, model, topk=5, device ='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    processed = process_image(image_path)
    # convert from np array to torch tensor. 
    processed = torch.from_numpy(processed).type(torch.FloatTensor)
    # Still had error. searching google lead here: https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    processed.unsqueeze_(0)
    # send to gpu 
    processed = processed.to(device)
    # send through model
    log_output = model.forward(processed)
    # get probabilities
    probs = torch.exp(log_output)
    # use topk. this was the weirdest solution. first issue was numpy being called on Variable requiring grad. 
    # so use detach. second issue was tensors still being cuda. send to cpu. finally, detach
    # broke hashability so added tolist()
    top_probs, top_labs = probs.topk(5)[0].detach().cpu().numpy().tolist()[0], probs.topk(5)[1].detach().cpu().numpy().tolist()[0]
    
    # inverse mapping to get indices back
    inv_map = {v: k for k, v in model.class_to_idx.items()}
    # get labels from indices
    top_labs = [inv_map[top_labs[x]] for x in range(len(top_labs))]
    return top_probs, top_labs

probs, classes = predict(data_dir + '/test/1/image_06743.jpg',vgg16)
print(probs)
print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[46]:


# TODO: Display an image along with the top 5 classes

def plot_solution(image_path, model):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    
    # Set up title
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]

    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);
    
    # Make prediction
    probs, labs = predict(image_path, model) 
    flowers = [cat_to_name[labs[i]] for i in range(len(labs))]
    
    # Plot bar chart
    # got ideas from https://pythonspot.com/matplotlib-bar-chart/
    y_pos = np.flipud(np.arange(len(flowers)))
    plt.subplot(2,1,2)
    plt.barh(y_pos,probs, align='center', alpha=0.5)
    plt.yticks(y_pos,flowers)
    plt.show()

image_path = data_dir + '/test/28/image_05230.jpg'
plot_solution(image_path, vgg16)

