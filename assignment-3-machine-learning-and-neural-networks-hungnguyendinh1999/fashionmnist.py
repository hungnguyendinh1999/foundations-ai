import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

'''
In this file you will write end-to-end code to train a neural network to categorize fashion-mnist data
'''


'''
PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted.
'''
# Use transforms to convert images to tensors and normalize them
transform = transforms.Compose([transforms.ToTensor()]) # no need to normalize
batch_size = 64

'''
PART 2:
Load the dataset. Make sure to utilize the transform and batch_size you wrote in the last section.
'''

trainset = torchvision.datasets.FashionMNIST("./data", download=True, 
                                             transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size)

testset = torchvision.datasets.FashionMNIST("./data", download=True, train=False, 
                                            transform=transform) 
testloader = torch.utils.data.DataLoader(testset, batch_size)

'''
PART 3:
Design a multi layer perceptron. Since this is a purely Feedforward network, you mustn't use any convolutional layers
Do not directly import or copy any existing models.
'''

count_classes = 10
# labels table is here https://github.com/zalandoresearch/fashion-mnist#labels
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(in_features=28*28, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=count_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
        
net = Net()

'''
PART 4:
Choose a good loss function and optimizer
'''

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

'''
PART 5:
Train your model!
'''

num_epochs = 30
training_loss_over_time = []
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Training loss: {running_loss}")
    training_loss_over_time.append(running_loss)

print('Finished Training')


'''
PART 6:
Evalute your model! Accuracy should be greater or equal to 80%

'''

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        inputs, labels = data # labels.shape = [64]
        outputs = net(inputs) # outputs.shape = [64, 10]
        _, pred_label = torch.max(outputs, dim = 1)
        correct += (pred_label == labels).sum().item()
        total += labels.size(0) # or batch_size


print('Accuracy: ', correct/total)

'''
PART 7:
Check the written portion. You need to generate some plots. 
'''

# 1. Figure examples of correct and incorrect prediction
correct_example = {}
incorrect_example = {}
with torch.no_grad():
    for data in testloader:
        inputs, labels = data # labels.shape = [64]
        outputs = net(inputs) # outputs.shape = [64, 10]
        _, pred_label = torch.max(outputs, dim = 1)
        bool_tensor = (pred_label != labels)
        
        if (bool_tensor.sum() > 0):
            # found incorrection
            inc_index = bool_tensor.nonzero()[0].item()
            incorrect_example = {
                "image": inputs[inc_index],
                "pred_label": pred_label[inc_index].item(),
                "actual_label": labels[inc_index].item()
            }
            # found correction
            corr_index = (~bool_tensor).nonzero()[0].item()
            correct_example = {
                "image": inputs[corr_index],
                "pred_label": pred_label[corr_index].item(),
                "actual_label": labels[corr_index].item()
            }
            break
# Mapping of label name
map_labels = {
    0   :"T-shirt/top",
    1	:"Trouser",
    2	:"Pullover",
    3	:"Dress",
    4	:"Coat",
    5	:"Sandal",
    6	:"Shirt",
    7	:"Sneaker",
    8	:"Bag",
    9	:"Ankle boot"
}

def saveImage(filename, tensor_image, required_text):
    fig, ax = plt.subplots()
    ax.imshow(tensor_image.permute(1, 2, 0), cmap='gray')
    ax.set_xlabel(required_text)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename)

# An incorrectly classified figure + clear label for predicted and true class
tensor_image = incorrect_example["image"]
required_text = "Predicted: {0}\nActual: {1}".format(
        map_labels[incorrect_example["pred_label"]], 
        map_labels[incorrect_example["actual_label"]])
saveImage("incorrect.png", tensor_image, required_text)

# Single Image correctly classified + clear label
tensor_image = correct_example["image"]
required_text = "Predicted: {0}".format(
    map_labels[correct_example["pred_label"]])
saveImage("correct.png", tensor_image, required_text)

# 2. Plot Training loss over time
plt.figure(figsize=(8,8))
plt.plot(range(len(training_loss_over_time)), training_loss_over_time)
plt.title("Training Loss over time")
plt.xlabel("Epochs")
plt.ylabel("Training Loss (Cross-Entropy)")
plt.savefig("loss_over_time.png")

# 3. Accuracy on testing dataset
# will take photo proof