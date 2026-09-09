
import numpy as np
import matplotlib.pyplot as plt
from Adaline import Adaline
from RBFNN import RBFNN

samples = 3000
hidden_units_list = [60, 80, 90, 100, 110, 120, 140]
epochs = 500
eta = 0.0005

X_train = np.random.uniform(-5, 5, (samples, 2))
Y_raw = np.sin(np.sqrt(X_train[:, 0]**2 + X_train[:, 1]**2)).reshape(-1, 1)
Y_min, Y_max = Y_raw.min(), Y_raw.max()
Y_train = 2 * (Y_raw - Y_min) / (Y_max - Y_min) - 1

best_mse = float('inf')
best_rbfnn = None
best_hidden = None

for hidden_neurons in hidden_units_list:
    print(f"Entrenando con {hidden_neurons} neuronas ocultas...")
    rbfnn = RBFNN(
        numberInputs=2,
        numberHideUnits=hidden_neurons,
        numberOutputs=1,
        eta=eta,
        output_estimator_class=Adaline,
        epochs=epochs
    )
    rbfnn.fit(X_train, Y_train)
    Y_pred = rbfnn.predict(X_train)
    mse = np.mean((Y_pred - Y_train.flatten()) ** 2)
    print(f"MSE: {mse:.6f}")
    if mse < best_mse:
        best_mse = mse
        best_rbfnn = rbfnn
        best_hidden = hidden_neurons

print(f"Mejor configuración: {best_hidden} neuronas ocultas con MSE = {best_mse:.6f}")

x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
X_flat = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
Z_real = np.sin(np.sqrt(X_flat[:, 0]**2 + X_flat[:, 1]**2)).reshape(X_grid.shape)
Z_pred_norm = best_rbfnn.predict(X_flat).reshape(X_grid.shape)
Z_pred = (Z_pred_norm + 1) / 2 * (Y_max - Y_min) + Y_min

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_grid, Y_grid, Z_real, cmap='coolwarm')
ax1.set_title("Función Real")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f(x, y)")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X_grid, Y_grid, Z_pred, cmap='coolwarm')
ax2.set_title(f"Mejor RBFNN ({best_hidden} neuronas ocultas)")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("f(x, y) estimada")

plt.tight_layout()
plt.show()

