import tarzan.core as tz
import torch
from torchvision import transforms


# User provides annotation file in .csv format (with header)
# First column is image name, second column is labels
# User provides all images in a dir
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
dataset.split()
# dataset.save(save_path)
# dataset.load(save_path)

train_dataloader = dataset.train_dataloader()
test_dataloader = dataset.test_dataloader()


# instantiate the neural network
clf = tz.Classifier("mlp")
clf.set_optimiser("adam")
clf.set_scheduler("exponential")

trainer = tz.Trainer()
trainer.fit(clf, train_dataloader, epochs=1)
# acc = trainer.test(test_dataloader)


# # save parameters of trained model
# PATH = "net_state/mlp.pth"
# trainer.save_net(PATH)


# mlp = tz.MLP()
# clf = tz.Classifier(mlp)
# # clf.set_optimiser("adam")
# # clf.set_scheduler("exponential")

# trainer = tz.Trainer()
# PATH = "net_state/mlp.pth"
# trainer.load_net(clf, PATH)


predictions = trainer.predict(test_dataloader)
trainer.count_classes(test_dataloader)

