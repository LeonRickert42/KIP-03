import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def loss_function(w1, w2, x1=1.0, x2=1.5, y_tar=2.0):
    if isinstance(w1, (int, float, np.float64)) and isinstance(w2, (int, float, np.float64)):
        return 0.5 * (np.sin(w1 * x1) + np.cos(w2 * x2) + w2 - y_tar) ** 2
    else:
        w1_sym, w2_sym = sp.symbols('w1 w2')
        error_expr = 0.5 * (sp.sin(w1_sym * x1) + sp.cos(w2_sym * x2) + w2_sym - y_tar) ** 2
        return error_expr.subs({w1_sym: w1, w2_sym: w2})


def gradient(w1, w2, x1=1.0, x2=1.5, y_tar=2.0):
    w1_sym, w2_sym = sp.symbols('w1 w2')
    error_expr = 0.5 * (sp.sin(w1_sym * x1) + sp.cos(w2_sym * x2) + w2_sym - y_tar) ** 2
    d_error_dw1 = sp.diff(error_expr, w1_sym)
    d_error_dw2 = sp.diff(error_expr, w2_sym)
    gradient_w1 = d_error_dw1.subs({w1_sym: w1, w2_sym: w2})
    gradient_w2 = d_error_dw2.subs({w1_sym: w1, w2_sym: w2})
    return gradient_w1, gradient_w2


def viz(current_w1, current_w2, current_loss):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    ax.plot(current_w1, current_w2, current_loss, marker="o", markersize=10, markeredgecolor="black",
            markerfacecolor="white")
    ax.set_title("Fehlergebirge", fontsize=13)
    ax.set_xlabel('w1', fontsize=11)
    ax.set_ylabel('w2', fontsize=11)
    ax.set_zlabel('E', fontsize=11)
    plt.savefig('figure.png')
    plt.close()


def gradient_descent(w1, w2, alpha, iterations, x1=1.0, x2=1.5, y_tar=2.0):
    for _ in range(iterations):
        gradient_w1, gradient_w2 = gradient(w1, w2, x1, x2, y_tar)
        w1 = w1 - alpha * gradient_w1
        w2 = w2 - alpha * gradient_w2
        loss = loss_function(w1, w2, x1, x2, y_tar)
        print(f"Iteration: {_ + 1}, w1: {w1:.4f}, w2: {w2:.4f}, Loss: {loss:.4f}")
    return w1, w2, loss


# Example runs
start_positions = [(-6.5, -9.5), (0.0, -0.5)]

for i, start_position in enumerate(start_positions, start=1):
    print(f"\nRun {i} with start position: {start_position}")
    current_w1, current_w2 = start_position
    current_loss = loss_function(current_w1, current_w2)
    viz(current_w1, current_w2, current_loss)
    # Gradient Descent
    final_w1, final_w2, final_loss = gradient_descent(current_w1, current_w2, alpha=0.05, iterations=100)

    print(f"Final values - w1: {final_w1:.4f}, w2: {final_w2:.4f}, Loss: {final_loss:.4f}")
    viz(final_w1, final_w2, final_loss)
