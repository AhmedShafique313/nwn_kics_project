import h5py
import numpy as np

# Load the .mat file
file_path = r'C:\Users\Personal\Documents\projects\KICS Second Project\conversion_data\ALLMICROGUNSHOTS.mat'
mat_data = h5py.File(file_path, 'r')

# Inspect the keys in the .mat file
print(list(mat_data.keys()))

# Access the data using the correct key
data = mat_data['noisemicroarray'][:]

# Print the shape of the data to confirm it's a 3D matrix
print(data.shape)

# Print the data to inspect it
print(data)

# Close the file
mat_data.close()
