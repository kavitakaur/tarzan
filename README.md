# Tarzan package

### The code has been divided into 3 main Classes - Data, Classifier and Trainer.

## Class: Data

The Data Class mainly encompasses the Data tasks of the problem statement in terms of preparation of data for the subsequent training. The anonymization step is done here.

## Class: Classifier

The Classifier Class serves to instantiate the option of the neural network to use by inputting the specific Class of neural network to use. 

The various neural networks are designed in their own individual Classes. For example the MLP and CNN Classes in the core.py file.

In this Classifier Class, users can also set the Learning Rate scheduler and Optimiser from the various options provided.

## Class: Trainer

The Trainer Class then takes in arguments that are derived from the Data and Classfier Classes, namely the prepared data and the instantiated neural network with its corresponding components. 

The dataset is trained here and the various machine learning tasks are also carried out here.


### Tutorial: Predicting images (with existing example data)

Import dependencies.

```
import tarzan.core as tz
import torch
from torchvision import transforms, datasets
```

User provides annotation file of images in .csv format (with header). First column of .csv file is image name, second column is labels. User to provide file path of directory containing all images. As an example, you could refer to the MNIST_jpg data which is provided in this repository.

Transformation of the data in terms of normalisation can also be done in this step.

```
annot_path = 'data/mnist_jpg/labels.csv'
mnist_path = 'data/mnist_jpg/raw'
save_path = 'data/mnist_jpg/processed'

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = tz.Data()
dataset.load_image(
    annot_path,
    mnist_path,
    transformations
)
```

The data can then be split into training and test sets with default proportion of 80% train & 20% test but user is allowed to change this by inputting a value between 0 and 1 - with 0.5 to mean 50%, etc.

This data could be saved and loaded to and from a selected path. This step is optional.

The next steps convert train and test dataset into dataloader format so that it can be run in the Trainer Class.

```
dataset.split()
# dataset.save(save_path)
# dataset.load(save_path)

train_dataloader = dataset.train_dataloader()
test_dataloader = dataset.test_dataloader()
```

User needs to instantiate a network under the Classifier Class with a string, either "mlp" or "cnn" (these are the 2 choices for now. More can be easily added). 

Under the Classifier Class is also where user selects the option for Optimiser (either "adam" or "sgd") and Learning Rate Scheduler (either "exponential" or "multistep", more options could also be added). 

Both the set_optimiser and set_scheduler methods require inputs to be in string form and a ValueError with the input options is raised should the user provide an incorrect input.

The trainer is then instantiated and set to run. User can change the epochs by inputting an int value, otherwise the default value is set at 5 at the moment.

```
# instantiate the neural network
clf = tz.Classifier("mlp")
clf.set_optimiser("adam")
clf.set_scheduler("exponential")

trainer = tz.Trainer()
trainer.fit(clf, train_dataloader, epochs=1)
# acc = trainer.test(test_dataloader)
```

Here we can run the machine learning tasks. What has been provided are for the character recognition and counting the number of different characters.

```
predictions = trainer.predict(test_dataloader)
trainer.count_classes(test_dataloader)
```


### Tutorial: To save parameters of trained model so that it can be loaded again

User provides the file path of directory for where to save the trained model. This option is available so that the user can go back to the trained model again to use it in testing for example, without having to carry out the whole training steps again. 

Without this step, user will have to retrain the data again if they wish to refer to it again after shutting down the program.

```
# save parameters of trained model
PATH = "net_state/mlp.pth"
trainer.save_net(PATH)
```

To load the trained parameters, user has to instantiate the corresponding neural network first. Ensure that the neural network instantiated has the same architecture as the one used in the training.

```
# Instantiate the nn used to train data
clf = tz.Classifier("mlp")
# clf.set_optimiser("adam")
# clf.set_scheduler("exponential")

trainer = tz.Trainer()
PATH = "net_state/mlp.pth"
trainer.load_net(clf, PATH)
```