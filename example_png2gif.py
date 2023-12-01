import imageio
import os

# Directory containing PNG images
png_directory = 'dataset/waic-tsr/water/ground_truth/water'

# List PNG files in the directory
png_files = sorted([os.path.join(png_directory, file) for file in os.listdir(png_directory) if file.endswith('.png')])

# Create a list to store images
images = []
for file in png_files:
    images.append(imageio.imread(file))  # Read each PNG image and append to the list

# Output GIF file name
output_gif = 'water.gif'

# Save the list of images as a GIF
imageio.mimsave(output_gif, images, format='GIF', fps=30)  # Adjust duration as needed (in seconds)

print(f"Created GIF: {output_gif}")
