import torch
from torch import optim
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def ackley(x, y):
    return -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) - torch.exp(0.5 * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))) + torch.exp(torch.tensor([1.0])) + 20


###############################################################################################
# Plot the objective function

# You will need to use Matplotlib's 3D plotting capabilities to plot the objective functions.
# Alternate plotting libraries are acceptable.
###############################################################################################


def plot3D(X, Y, Z, plot_name, title=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:3g}')
    # ax.zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style="sci", axis="z", scilimits=(0,3), useOffset=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if not title:
        title = "Surface Map of "+ plot_name + " function"
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(plot_name+".png", format='png')

def make_data(func, low=-30, high=30, step=0.5, require_tensor=False):
    # Make up data
    X = np.arange(low, high, step)
    Y = np.arange(low, high, step)
    X, Y = np.meshgrid(X, Y)
    if not require_tensor:
        Z = func(X,Y)
    else:
        x = torch.tensor(X)
        y = torch.tensor(Y)
        Z = func(x,y)
    return X, Y, Z

X, Y, Z = make_data(rosenbrock)
plot3D(X, Y, Z, "rosenbrock")

X, Y, Z = make_data(beale)
plot3D(X, Y, Z, "beale")

X, Y, Z = make_data(ackley, require_tensor=True)
plot3D(X, Y, Z, "ackley")

X, Y, Z = make_data(ackley, low=-3, high=3, require_tensor=True)
plot3D(X, Y, Z, "ackley3", title="Surface Map of ackley function in range [-3,3]")

###############################################################################################
# STOCHASTIC GRADIENT DESCENT

# Initialize x and y to 10.0 (ensure you set requires_grad=True when converting to tensor)

# Use Stochastic Gradient Descent in Pytorch to optimize the objective function.

# Save the values of the objective function over 5000 iterations in a list.

# Print the values of x, y, and the objective function after optimization.
###############################################################################################

def optimize_SGD(f, x, y, lr, iteration, debug=False):
    list_var = [f(x, y).item()]
    optimizer = torch.optim.SGD([x, y], lr=lr)

    for i in range(iteration):
        optimizer.zero_grad()
        f(x, y).backward()
        optimizer.step()
        list_var.append(f(x, y).item())

        if (debug) and ((i+1)%2000==0):
            print(f"============ {i+1} ===============")
            print("The result of f(x, y) is: ", f(x, y).item())
            print("The values of x and y are: ", x.item(), y.item())
        if np.isnan(f(x, y).item()):
            print("Terminate on nan!!! @ iteration ", i)
            break
                
    return x, y, list_var
map_f_to_iterations = {
    ackley: 5000,
    beale: 10000, 
    rosenbrock: 5000
}

map_f_to_lr_sgd = {
    ackley: 0.131,
    beale: 0.0000001, # done
    rosenbrock: 5.5687e-05 # done
}
print("\n================= S G D ======================")

dict_of_list_var_sgd = {}
for f, lr in map_f_to_lr_sgd.items():
    x, y = 10.0, 10.0
    x = torch.tensor([x], requires_grad=True)
    y = torch.tensor([y], requires_grad=True)
    x,y, list_var_sgd = optimize_SGD(f, x, y, lr, map_f_to_iterations[f])
    dict_of_list_var_sgd[f] = list_var_sgd
    
    print(f"The result of {f.__name__}(x, y) is: ", f(x, y).item())
    print(f"The values of x and y are: ", x.item(), y.item())

###############################################################################################
# Adam Optimizer

# Re-initialize x and y to 10.0 (ensure you set requires_grad=True when converting to tensor)

# Use the Adam optimizer in Pytorch to optimize the objective function.

# Save the values of the objective function over 5000 iterations in a list.

# Print the values of x, y, and the objective function after optimization.
###############################################################################################

def optimize_adam(f, x, y, lr, iteration, debug=False):
    list_var = [f(x,y).item()]
    optimizer = torch.optim.Adam([x, y], lr=lr)

    for i in range(iteration):
        optimizer.zero_grad()
        f(x, y).backward()
        optimizer.step()
        list_var.append(f(x, y).item())
        if (debug) and ((i+1)%10000==0):
            print(f"============ {i+1} ===============")
            print("The result of f(x, y) is: ", f(x, y).item())
            print("The values of x and y are: ", x.item(), y.item())
        if np.isnan(f(x, y).item()):
            print("Terminate on nan!!! @ iteration ", i)
            break
                
    return x, y, list_var

print("\n================= A D A M ======================")
map_f_to_lr_adam = {
    ackley: 0.1,
    beale: 0.1, 
    rosenbrock: 0.1
}
dict_of_list_var_adam = {}
for f, lr in map_f_to_lr_adam.items():
    x, y = 10.0, 10.0
    x = torch.tensor([x], requires_grad=True)
    y = torch.tensor([y], requires_grad=True)
    x,y, list_var_adam = optimize_adam(f, x, y, lr, map_f_to_iterations[f], debug=False)
    dict_of_list_var_adam[f] = list_var_adam
    
    print(f"The result of {f.__name__}(x, y) is: ", f(x, y).item())
    print(f"The values of x and y are: ", x.item(), y.item())


###############################################################################################
# Comparing convergence rates

# Plot the previously stored values of the objective function over 5000 iterations for both SGD and Adam in a single plot.
###############################################################################################

for f, _ in map_f_to_lr_adam.items():
    plt.figure(figsize=(12,8))
    # SGD
    y_sgd = dict_of_list_var_sgd[f]
    plt.plot(range(len(y_sgd)), y_sgd, label=f"SGD lr={map_f_to_lr_sgd[f]}", linewidth=6.9)
    # ADAM
    y_adam = dict_of_list_var_adam[f]
    plt.plot(range(len(y_adam)), y_adam, label=f"ADAM lr={map_f_to_lr_adam[f]}", linewidth=6.9)
    plt.title(f"SGD vs. ADAM: Comparing Convergence Rates for {f.__name__}")
    plt.xlabel("Epoch (Iterations)")
    plt.ylabel(f"Output of {f.__name__}")
    plt.legend()
    plt.savefig(f"SGD_vs_ADAM_{f.__name__}.png")
    # Zoomed-in version
    plt.ylim((0, 20))
    plt.title(f"(ZOOMED IN) SGD vs. ADAM: Comparing Convergence Rates for {f.__name__}")
    plt.savefig(f"Zoom_SGD_vs_ADAM_{f.__name__}.png")