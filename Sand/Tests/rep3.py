

import numpy as np
import matplotlib.pyplot as plt


# Absorption trace (hollow triangles)
x_abs = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60,
                  0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30, 1.35, 1.40,
                  1.45, 1.50], dtype=float)
y_abs = np.array([0.8, 1.5, 2.3, 3.0, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
                  3.8, 3.9, 4.0, 4.1, 4.3, 4.5, 5.0, 5.8, 7.0, 8.5,
                  10.5, 13.0], dtype=float)

# Desorption trace (filled triangles)
x_des = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60,
                  0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30, 1.35, 1.40,
                  1.45, 1.50], dtype=float)
y_des = np.array([0.7, 1.4, 2.2, 2.9, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6,
                  3.7, 3.8, 3.9, 4.0, 4.2, 4.4, 4.8, 5.5, 6.5, 8.0,
                  10.0, 12.5], dtype=float)





# Create the plot
plt.figure(figsize=(16, 8))
plt.plot(x_abs, y_abs, marker='^', linestyle='-', color='g', markerfacecolor='none', markeredgecolor='blue', label='Absorption')
plt.plot(x_des, y_des, marker='^', linestyle='-', color='c', label='Desorption')

# Set logarithmic scale for y-axis
plt.yscale('log')

# Customize the plot
plt.xlabel('C [H] (wt.%)')
plt.ylabel('P [H2] (bar)')
plt.title('Hydrogen Pressure vs Hydrogen Concentration')
plt.grid(True, which="both", ls="--")
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()






