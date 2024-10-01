import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Author: Aman Hogan
# Visualizes results from goto_van.c

# Load the CSV file into a DataFrame
df = pd.read_csv("../output/gotovan.csv")

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['kc/mc'], df['nr'], df['mr'], c=df['gflops'], cmap='coolwarm', s=100)
plt.colorbar(sc, ax=ax, label='GFLOPS')

ax.set_xlabel('kc/mc')
ax.set_ylabel('nr')
ax.set_zlabel('mr')

# Set title
plt.title('3D Relationship between kc/mc, nr, mr, and GFLOPS')
ax.view_init(elev=10, azim=120)  # Adjust the elevation and azimuthal angle for rotation
ax.set_title('3D Scatter Plot of GFLOPS vs kc/mc, nr, and mr')
plt.savefig('../output/gotovan_heatmap.png')
