from os import X_OK
import  numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def create_dataset(w_star, x_range, sample_size, sigma, seed=None):
    random_state = np.random.RandomState(seed)
    x = random_state.uniform(x_range[0], x_range[1], (sample_size))
    X = np.zeros((sample_size, w_star.shape[0]))

    for i in range(sample_size):
        X[i, 0] = 1
        for j in range(1, w_star.shape[0]):
            X[i, j] = pow(x[i], j)

    y = X.dot(w_star)

    if sigma > 0:
        y += random_state.normal(0.0, sigma, sample_size)

    return X, y

sample_size = 100
w = np.array([-8, -4, 2, 1])
w_star = np.transpose(w)
X, y = create_dataset(w_star, [-3, 2], sample_size, 0.5, 0)
print("X")
print(X.shape)
X_val, y_val = create_dataset(w_star, [-3, 2], 100, 0.5, 1)

#plot the points
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.scatter(X[:, 1], y, color='r')
ax.scatter(X_val[:, 1], y_val, color='b')
ax.set_title('Points')
#plt.show()

#Linear regression model
model = nn.Linear(4, 1) # input dimension 1, and output dimension 1.
loss_fn = nn.MSELoss()
learning_rate = 0.00661
learning_rate = 0.00661
optimizer = optim.SGD(model. parameters (), lr= learning_rate )

#Reshape the input
X = X.reshape(sample_size, 4)
X = torch.from_numpy(X)
X = X.float()
y = torch.from_numpy (y.reshape(( sample_size , 1))).float()
X_val = torch.from_numpy(X_val.reshape(( sample_size , 4))).float()
y_val = torch.from_numpy(y_val.reshape(( sample_size , 1))).float()

#Linear regression

num_steps = 100

for step in range(num_steps):
    model.train()
    optimizer.zero_grad()

    y_ = model(X)
    loss = loss_fn(y_, y)
    print(f"Step: {step} train loss: {loss}")

    loss.backward()
    optimizer.step()

    #Evaluate model on validation set
    model.eval()
    with torch.no_grad():
        y_ = model(X_val)
        val_loss = loss_fn(y_, y_val)
    print(f"Step: {step} validation loss: {val_loss}")

print(f"Learning rate: {learning_rate}")

