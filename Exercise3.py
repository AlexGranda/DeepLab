import numpy as np
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

    actual_X = X[:, 0:4]
    y = actual_X.dot(w_star[0:4])

    if sigma > 0:
        y += random_state.normal(0.0, sigma, sample_size)

    return X, y


def generate_point_polynomial_matrix(x_matrix, x, sample_size, polynomial_size):
    for i in range(sample_size):
        x_matrix[i, 0] = 1
        for j in range(1, polynomial_size):
            x_matrix[i, j] = pow(x[i], j)


def plot_loss(title, training_loss, validation_loss, epochs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(color='0.95')
    ax.plot(epochs, training_loss, 'b', label='Training loss')
    ax.plot(epochs, validation_loss, 'r', label='Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    plt.show()


def scatter_points(X, X_val, y, y_val):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(color='0.95')
    ax.scatter(X[:, 1], y, color='r', label='Training data')
    ax.scatter(X_val[:, 1], y_val, color='b', label='Validation data')
    ax.set_title('Training and validation datasets')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()


def plot_polynomials(DEVICE, input_array, model, polynomial_size):
    point_number = 1000
    x = np.linspace(-3, 2, point_number)
    xm = np.zeros((point_number, polynomial_size))
    generate_point_polynomial_matrix(xm, x, point_number, polynomial_size)
    xm = torch.from_numpy(xm)
    xm = xm.float()
    xm = xm.to(DEVICE)
    model.eval()
    with torch.no_grad():
        y_ = model(xm)
    fig, ax = plt.subplots()
    original_p = np.poly1d(np.flipud(input_array[0:4]))  # Should invert the order of the vector
    evaluated_original_polynomial = original_p(x)
    ax.plot(x, evaluated_original_polynomial, ".", label='Original polynomial')
    ax.plot(x, y_, ".", label='Interpolated polynomial')
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.set_title('Interpolated polynomial VS Original Polynomial')
    ax.legend()
    plt.show()


def linear_regression_model(DEVICE, input_array, degree):
    training_sample_size = 100
    validation_sample_size = 100
    w = np.array(input_array)
    w_star = np.transpose(w)
    sigma = 0.5
    polynomial_size = degree + 1

    # Generate the training and validation datasets with the given parameters
    X, y = create_dataset(w_star, [-3, 2], training_sample_size, sigma, 0)
    X_val, y_val = create_dataset(w_star, [-3, 2], validation_sample_size, sigma, 1)

    # Plot the points
    scatter_points(X, X_val, y, y_val)

    # Reshape the input
    X = X.reshape(training_sample_size, polynomial_size)
    X = torch.from_numpy(X)
    X = X.float()
    X = X.to(DEVICE)
    y = torch.from_numpy(y.reshape((training_sample_size, 1))).float().to(DEVICE)
    X_val = torch.from_numpy(X_val.reshape((validation_sample_size, polynomial_size))).float().to(DEVICE)
    y_val = torch.from_numpy(y_val.reshape((validation_sample_size, 1))).float().to(DEVICE)

    model = nn.Linear(polynomial_size, 1, bias=False)  # input dimension 4, and output dimension 1.
    model = model.to(DEVICE)
    loss_fn = nn.MSELoss()
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Linear regression
    num_steps = 700
    training_loss = []
    validation_loss = []
    epochs = []

    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()

        y_ = model(X)
        loss = loss_fn(y_, y)
        training_loss.append(loss.detach())
        if step > 650:
            print(f"Step: {step} train loss: {loss}")

        loss.backward()
        optimizer.step()

        # Evaluate model on validation set
        model.eval()
        with torch.no_grad():
            y_ = model(X_val)
            val_loss = loss_fn(y_, y_val)
            validation_loss.append(val_loss.detach())

        if step > 650:
            print(f"Step: {step} validation loss: {val_loss}")

        epochs.append(step)

    print(f"Learning rate: {learning_rate}")
    print(model.weight)

    # Plot training loss
    plot_loss('Training Loss', training_loss, validation_loss, epochs)

    # Get the prediction from the final model.
    plot_polynomials(DEVICE, input_array, model, polynomial_size)


# Declare the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Declaration of the linear regression model
linear_regression_model(DEVICE, [-8, -4, 2, 1], 3)
# linear_regression_model(DEVICE, [-8, -4, 2, 1, 1], 4)
