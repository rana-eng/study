
import matplotlib.pyplot as plt
import pandas as pd
#print("Pandas is being loaded from:", pd.__file__)

# Load the dataset
df = pd.read_csv(r'D:\Study\digit_recognizer\data\train.csv')
print(df.head())
X = df

# Separate labels and pixels
y = df['label']                   # Extract only the target column
X = df.drop(columns=['label'])    # Drop the target column to keep only pixel data

# Verify the shapes
print("X shape (pixels):", X.shape)
print("y shape (labels):", y.shape)

# .values extracts the raw numbers without the column headers
single_row = X.iloc[784].values

# 2. Reshape the 784 pixel values into a 28x28 matrix
digit_image = single_row.reshape(28, 28)

# 3. Plot the matrix as an image
plt.imshow(digit_image, cmap='gray')
plt.title("Inspecting Example Index 5")
plt.axis('off')  # Hide the pixel grid coordinate numbers
plt.show()

# Find the minimum and maximum pixel values
min_pixel = X.min().min()
max_pixel = X.max().max()

print(f"Minimum pixel value: {min_pixel}")
print(f"Maximum pixel value: {max_pixel}")

# Scale all pixel values to the range 0.0 - 1.0
X_scaled = X / 255.0

# Verify the new range
print(f"New Minimum pixel value: {X_scaled.min().min()}")
print(f"New Maximum pixel value: {X_scaled.max().max()}")
