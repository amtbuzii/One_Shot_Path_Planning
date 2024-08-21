import matplotlib.pyplot as plt
import pandas as pd
result_csv_file = '100x100_threshold_results.csv'
threshold_results_df = pd.read_csv(result_csv_file)
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_results_df.index = THRESHOLDS

# Plotting paths_found and paths_not_found
plt.figure(figsize=(12, 6))

# Plot paths_found
plt.plot(threshold_results_df.index, threshold_results_df['paths_found'], marker='o', linestyle='-', color='b', label='Paths Found')

# Plot paths_not_found
#plt.plot(threshold_results_df.index, threshold_results_df['paths_not_found'], marker='o', linestyle='-', color='g', label='Paths Not Found')

# Add labels and title
plt.xlabel('Threshold')
plt.ylabel('%')
#plt.title('Paths Found vs. Paths Not Found')
#plt.legend()

# Show the plot
#plt.grid(True)
#plt.show()

# Plotting paths_are_equal and paths_are_longer
#plt.figure(figsize=(12, 6))

# Plot paths_are_equal
plt.plot(threshold_results_df.index, threshold_results_df['paths_are_equal'], marker='o', linestyle='-', color='r', label='Paths Are Equal')

# Plot paths_are_longer
#plt.plot(threshold_results_df.index, threshold_results_df['paths_are_longer'], marker='o', linestyle='-', color='purple', label='Paths Are Longer')

# Add labels and title
plt.xlabel('Threshold')
#plt.ylabel('Count - %')
plt.title('100X100 test')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# Plotting prediction time
plt.figure(figsize=(12, 6))

# Plot paths_found
plt.plot(threshold_results_df.index, threshold_results_df['prediction_time'], marker='o', linestyle='-', color='b', label='prediction_time')

# Plot paths_not_found
#plt.plot(threshold_results_df.index, threshold_results_df['paths_not_found'], marker='o', linestyle='-', color='g', label='Paths Not Found')

# Add labels and title
plt.xlabel('Threshold')
plt.ylabel('ms')
plt.grid(True)
plt.show()