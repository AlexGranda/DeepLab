"""Assignment2_Granda_Alexandra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

  def imshow(train_set):
      fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
      image_1 = np.squeeze(train_set[3][0].numpy())
      image_2 = np.squeeze(train_set[12][0].numpy())
      image_3 = np.squeeze(train_set[2][0].numpy())
      ax1.imshow(np.moveaxis(image_1, 0, -1)) 
      ax2.imshow(np.moveaxis(image_2, 0, -1)) 
      ax3.imshow(np.moveaxis(image_3, 0, -1)) 

      plt.show()

  def charge_images(batch_size):
      train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
      train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

      #Plot one image
      print(f'The number of images in the dataset is: {len(train_set)}')
      imshow(train_set)

      #Calculate means and standard devation
      means = calculate_means(len(train_set), train_loader)
      std = calculate_deviations(len(train_set), train_loader, means)

      transf = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(means, std),
      ])

      train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transf, download=True)

      # Split the training data into training and validation parts.
      # here we will use `torch.utils.data.SubsetRandomSampler`.

      idx = np.arange(len(train_set))

      # Use last 1000 images for validation
      val_indices = idx[50000-1000:]
      train_indices= idx[:-1000]

      train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
      valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

      train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                sampler=train_sampler, num_workers=2)

      valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=2)
      
      return train_loader, valid_loader

  def trainConvNet(num_epochs=28, batch_size=32, leaning_rate=0.002, momentum=0.9, dropout=True, p=0.4):
      device = torch.device("cuda:0" if torch.cuda. is_available () else "cpu")
      # Hyper−parameters to specify model:
      hidden_size = 512
      num_classes = 10
      p = 0.4

      # for training:
      num_epochs = num_epochs
      batch_size = batch_size
      learning_rate = leaning_rate
      momentum = momentum
      
      train_loader, valid_loader = charge_images(batch_size)
      
      if not dropout:
        convNet = ConvNet(hidden_size, num_classes)
      else:
        convNet = ConvNetWithDropout(hidden_size, num_classes, p)
      convNet.to(device)

      # Create loss and optimizer
      loss_fn = nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(convNet.parameters(), lr=learning_rate, momentum=momentum)
      print(convNet)

      loss_vals = []
      training_accuracy = []

      validation_loss_vals = []
      validation_accuracy = []

      for epoch in range(1, num_epochs):
          running_loss = 0.0
          running_total = 0
          running_total_sum = 0
          running_correct = 0
          running_correct_sum = 0
          run_step = 0
          epoch_loss = []
          epoch_validation_loss = []
          for i, (images, labels) in enumerate(train_loader):
              convNet.train() 
              images = images.to(device)
              labels = labels.to(device)  # shape (B).
              outputs = convNet(images)  # shape (B, 10).
              loss = loss_fn(outputs, labels)
              optimizer.zero_grad()  # reset gradients.
              loss.backward()  # compute gradients.
              optimizer.step()  # update parameters.

              epoch_loss.append(loss.item())
              running_loss += loss.item()
              running_total += labels.size(0)

              with torch.no_grad():
                  _, predicted = outputs.max(1)
              running_correct += (predicted == labels).sum().item()
              run_step += 1
              if i % 200 == 0:
                  # check accuracy.
                  print(f'epoch: {epoch}, steps: {i}, '
                        f'train_loss: {running_loss / run_step :.3f}, '
                        f'running_acc: {100 * running_correct / running_total:.1f} %')
                  running_correct_sum += running_correct
                  running_total_sum += running_total
                  running_loss = 0.0
                  running_total = 0
                  running_correct = 0
                  run_step = 0
          
          loss_vals.append(sum(epoch_loss)/len(epoch_loss))
          training_accuracy.append((100 * running_correct_sum)/running_total_sum)

          # validate
          correct = 0
          total = 0
          convNet.eval()
          with torch.no_grad():
              for data in valid_loader:
                  images, labels = data
                  images, labels = images.to(device), labels.to(device)
                  outputs = convNet(images)
                  loss = loss_fn(outputs, labels)
                  epoch_validation_loss.append(loss.item())
                  _, predicted = outputs.max(1)
                  total += labels.size(0)
                  correct += (predicted == labels).sum().item()
                  val_acc = 100 * correct / total
          validation_loss_vals.append(sum(epoch_validation_loss)/len(epoch_validation_loss))
          validation_accuracy.append(100 * correct / total)
          print(f'Validation accuracy: {100 * correct / total} %')
          print(f'Validation error rate: {100 - 100 * correct / total: .2f} %')
      
      best_val_accuracy = max(validation_accuracy)
      print(f'Best validation value was: {best_val_accuracy}%, at epoch: {validation_accuracy.index(best_val_accuracy) + 1}')

      # Plot loss evolution
      fig = plt.figure()
      ax = fig.add_subplot(111)
      plt.grid(color='0.95')
      ax.plot(np.linspace(1, num_epochs-1, num_epochs-1).astype(int), loss_vals, 'b', label='Training loss')
      ax.plot(np.linspace(1, num_epochs-1, num_epochs-1).astype(int), validation_loss_vals, 'r', label='Validation loss')
      ax.set_xlabel('Epochs')
      ax.set_ylabel('Loss')
      ax.set_title('Training and validating loss')
      ax.legend()
      plt.show()

      # Plot accuracy evolution
      fig = plt.figure()
      ax = fig.add_subplot(111)
      plt.grid(color='0.95')
      ax.plot(np.linspace(1, num_epochs-1, num_epochs-1).astype(int), training_accuracy, 'b', label='Training accuracy')
      ax.plot(np.linspace(1, num_epochs-1, num_epochs-1).astype(int), validation_accuracy, 'r', label='Validation accuracy')
      ax.set_xlabel('Epochs')
      ax.set_ylabel('Accuracy')
      ax.set_title('Training and validating accuracy')
      ax.legend()
      plt.show()

      print('Finished Training')
      torch.save(convNet.state_dict(), 'basic_model.pt')

  def calculate_deviations(train_set_length, train_loader, means):
    pixels = train_set_length * 32 * 32
    squared_error_sum = np.zeros(3)

    for i in range(3):
      for batch in train_loader:
        difference = batch[0][:,i] - means[i]
        squared_difference = np.square(difference)
        squared_error_sum[i] += squared_difference.sum()

    standard_deviation = torch.sqrt(torch.from_numpy(squared_error_sum) / (pixels - 1))

    return standard_deviation

  def calculate_means(train_set_length, train_loader):
    pixels = train_set_length * 32 * 32
    sum = np.zeros(3)

    for i in range(3):
      for batch in train_loader:
        sum[i] += batch[0][:,i].sum()

    means = sum / pixels

    return means

  class ConvNet(nn.Module):
      def __init__(self, hidden_size, num_classes):
          super(ConvNet, self).__init__()
          self.conv1 = nn.Conv2d(3, 32, 3)
          self.conv2 = nn.Conv2d(32, 32, 3)
          self.conv3 = nn.Conv2d(32, 64, 3)
          self.conv4 = nn.Conv2d(64, 64, 3)
          self.pool = nn.MaxPool2d(2, 2)
          self.fc1 = nn.Linear(64 * 5 * 5, hidden_size)
          self.fc2 = nn.Linear(hidden_size, num_classes)

      def forward(self, x):
          x = F.relu(self.conv1(x))
          x = self.pool(F.relu(self.conv2(x)))
          x = F.relu(self.conv3(x))
          x = self.pool(F.relu(self.conv4(x)))
          x = x.view(-1, 64 * 5 * 5)
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return x

  class ConvNetWithDropout(nn.Module):
      def __init__(self, hidden_size, num_classes, p=0.5):
          super(ConvNetWithDropout, self).__init__()
          self.conv1 = nn.Conv2d(3, 32, 3)
          self.conv2 = nn.Conv2d(32, 32, 3)
          self.conv3 = nn.Conv2d(32, 64, 3)
          self.conv4 = nn.Conv2d(64, 64, 3)
          self.pool = nn.MaxPool2d(2, 2)
          self.dropout = nn.Dropout(p)
          self.fc1 = nn.Linear(64 * 5 * 5, hidden_size)
          self.fc2 = nn.Linear(hidden_size, num_classes)

      def forward(self, x):
          x = F.relu(self.conv1(x))
          x = self.pool(F.relu(self.conv2(x)))
          x = self.dropout(x)
          x = F.relu(self.conv3(x))
          x = self.pool(F.relu(self.conv4(x)))
          x = self.dropout(x)
          x = x.view(-1, 64 * 5 * 5)
          x = F.relu(self.fc1(x))
          x = self.dropout(x)
          x = self.fc2(x)
          return x

  def test_model(batch_size=32, dropout=True, p=0.4):
    device = torch.device("cuda:0" if torch.cuda. is_available () else "cpu")
    correct = 0
    total = 0
    batch_size = batch_size

    hidden_size = 512
    num_classes = 10
    p = p

    loss_fn = nn.CrossEntropyLoss()

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    means = calculate_means(len(test_data), test_loader)
    std = calculate_deviations(len(test_data), test_loader, means)

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, std),
    ])

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transf, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    loaded_dict = torch.load('basic_model.pt')

    if not dropout:
      convNet = ConvNet(hidden_size, num_classes)
    else:
      convNet = ConvNetWithDropout(hidden_size, num_classes, p)

    convNet.load_state_dict(loaded_dict)
    convNet.to(device)
    convNet.eval()

    test_loss_vals = []
    test_accuracy = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = convNet(images)
            loss = loss_fn(outputs, labels)
            test_loss_vals.append(loss.item())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = sum(test_loss_vals)/len(test_loss_vals)
    test_accuracy = 100 * correct / total
    print(f'Test accuracy: {test_accuracy} %')
    print(f'Test error rate: {100 - test_accuracy: .2f} %')
    print(f'Test loss: {test_loss: .2f} %')

  # Uncomment following line to train CNN without dropout
  # trainConvNet(num_epochs=24, batch_size=32, leaning_rate=0.002, momentum=0.9, dropout=False)
  trainConvNet(num_epochs=28, batch_size=32, leaning_rate=0.002, momentum=0.9, dropout=True, p=0.4)

  test_model(batch_size=32, dropout=True, p=0.4)