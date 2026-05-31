import numpy as np

# 1. Create a dummy array with a fixed size of 6 elements, including NaNs
similarity_history = np.array([0.85, np.nan, 0.92, np.nan, 0.78, 0.95])
print("Original history (fixed-size with NaNs):")
print(similarity_history)
print(f"Shape: {similarity_history.shape}\n")

# 2. Your exact code snippet to remove NaNs
similarity_history = similarity_history[~np.isnan(similarity_history)]

# 3. View the cleaned results
print("Cleaned history (NaNs removed):")
print(similarity_history)
print(f"Shape: {similarity_history.shape}")
