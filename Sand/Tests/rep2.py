
import numpy as np
import matplotlib.pyplot as plt


# Data

# X-axis: C [H] (wt.%)
x_axis = np.array([0.08, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 
                   0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30, 1.35, 
                   1.40, 1.45, 1.50], dtype=float)

# Y-axis: P [H2] (bar) - logarithmic scale
y_axis = np.array([0.8, 1.2, 2.0, 2.5, 3.0, 3.5, 3.8, 3.9, 4.0, 4.0, 
                   4.0, 4.0, 4.1, 4.2, 4.3, 4.5, 4.8, 5.5, 6.5, 7.8, 
                   9.5, 11.5, 13.5], dtype=float)


# ---


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_axis, y_axis, marker='o', linestyle='-', color='b', label='P [H2] vs C [H]')

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











