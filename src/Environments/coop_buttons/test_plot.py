import numpy as np
import matplotlib.pyplot as plt

# Given grid
grid = np.array([[1, 1, 9, -1, 0, 2, 2, -1, 3, 3],
                 [0, 1, 0, -1, 0, 0, 2, -1, 0, 3],
                 [0, 0, 0, -1, 8, 8, 8, -1, 8, 8],
                 [0, 0, 0, -1, 8, 8, 8, -1, 8, 8],
                 [0, 0, 0, -1, 0, 0, 0, -1, 0, 0],
                 [0, 0, 0, -1, 0, 0, 9, 0, 0, 0],
                 [0, 0, 0, -1, 0, 0, 0, 0, 0, 9],
                 [0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                 [0, 0, 0, 0, 0, 8, 8, 8, 8, 9],
                 [0, 0, 0, 0, 0, 8, 8, 8, 8, 0]])

colors = {-1: 'black', 0: 'white', 8: 'gray', 9: 'yellow'}

# Add colors for agents (1, 2, 3)
for agent_id in range(1, 4):
    colors[agent_id] = 'red'

# Create a figure and axis
fig, ax = plt.subplots()

# Iterate through the grid and plot elements
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        # Plot rectangles with corresponding colors
        ax.add_patch(plt.Rectangle((j, -i - 1), 1, 1, color=colors[grid[i, j]]))

        # Plot circles for agent locations
        if grid[i, j] in [1, 2, 3]:
            ax.add_patch(plt.Circle((j + 0.5, -i - 0.5), 0.3, color='red'))

# Set x and y limits based on grid dimensions
ax.set_xlim(0, grid.shape[1])
ax.set_ylim(-grid.shape[0], 0)

# Remove x and y ticks and labels for cleaner visualization
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# Show the plot
plt.show()
