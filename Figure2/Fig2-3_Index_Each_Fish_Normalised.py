
from scipy.io import loadmat
from scipy.signal import, peak_widths
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

#Convert radians to degrees, create a new array and then find peaks in the original histogram
folder_path ='pathnamehere'
file_names = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
fitted_osi_angle_deg_mod = {}

for idx, file_name in enumerate(file_names, start=1):
    file_path = os.path.join(folder_path, file_name)
    data = loadmat(file_path)
    variable = data['DEFAULTS_POST_GROUP']
    fitted_osi_angle_deg = np.rad2deg(variable['fitted_osi_angle'][0, 0]) % 180
    fitted_osi_angle_deg_mod[idx] = fitted_osi_angle_deg
modified_fitted_osi_angle_deg_mod = {}


for idx, angles in fitted_osi_angle_deg_mod.items():
    mask = (angles >= 0) & (angles <= 25)
    modified_angles = np.where(mask, angles + 180, angles)
    modified_fitted_osi_angle_deg_mod[idx] = modified_angles


# Choose an example index (e.g., array 1)
example_idx = 2
original_angles = fitted_osi_angle_deg_mod[example_idx]
modified_angles = modified_fitted_osi_angle_deg_mod[example_idx]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
counts_orig, bins_orig = np.histogram(original_angles, bins=180, range=(0, 180))
axs[0].hist(original_angles, bins=180, color='blue', alpha=0.7)
axs[0].set_title(f"Original Array (Idx {example_idx})")
axs[0].set_xlabel("Degree")
axs[0].set_ylabel("Frequency")
axs[0].set_xlim([0, 180])

bin_centers_orig = (bins_orig[:-1] + bins_orig[1:]) / 2
peaks_orig, _ = find_peaks(counts_orig, distance=50)
peak_bins_orig = bin_centers_orig[peaks_orig]
axs[0].plot(peak_bins_orig, counts_orig[peaks_orig], "x", color='red', label='Peaks')
axs[0].legend()

counts_mod, bins_mod = np.histogram(modified_angles, bins=180, range=(25, 205))
axs[1].hist(modified_angles, bins=180, color='green', alpha=0.7)
axs[1].set_title(f"Modified Array (Idx {example_idx})")
axs[1].set_xlabel("Degree")
axs[1].set_ylabel("Frequency")
axs[1].set_xlim([25, 205])


bin_centers_mod = (bins_mod[:-1] + bins_mod[1:]) / 2
peaks_mod, _ = find_peaks(counts_mod, distance=50)
peak_bins_mod = bin_centers_mod[peaks_mod]
axs[1].plot(peak_bins_mod, counts_mod[peaks_mod], "x", color='red', label='Peaks')
axs[1].legend()
plt.tight_layout()
plt.show()


#convert arrays shape into histogram like (column 1: degree; column 2: number of bins)
#normalise the peaks by dividing each peak by the sum of all voxels; peaks are calculate with distance = 30
#find peaks and store into separate dictionary 


histogram_arrays = {}
normalized_histogram_arrays = {}
peak_data_list = []

rows, cols = 4, 5
fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()



for idx, (data, ax) in enumerate(zip(modified_fitted_osi_angle_deg_mod.values(), axes), start=1):
    hist, bin_edges = np.histogram(data, bins=180)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_rounded = np.round(hist).astype(int)
    bin_centers_rounded = np.round(bin_centers).astype(int)
    histogram_data = np.vstack((hist_rounded, bin_centers_rounded)).T
    df_histogram = pd.DataFrame(histogram_data, columns=['Frequency', 'Degree'])
    histogram_arrays[idx] = df_histogram
    total_frequency = np.sum(hist_rounded)
    if total_frequency > 0:
        normalized_hist = hist_rounded / total_frequency
    else:
        normalized_hist = hist_rounded

    normalized_histogram_data = np.vstack((normalized_hist, bin_centers_rounded)).T
    df_normalized_histogram = pd.DataFrame(normalized_histogram_data, columns=['Normalized Frequency', 'Degree'])
    normalized_histogram_arrays[idx] = df_normalized_histogram


    ax.bar(bin_centers_rounded, normalized_hist, width=1, alpha=0.7)
    ax.set_title(f"Animal {idx}")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Normalized Voxel Frequency")
    ax.set_xlim([25, 205])
    ax.set_ylim([0, 0.1])
    peaks, properties = find_peaks(normalized_hist, distance=25, height=0)
    peak_widths_result = peak_widths(normalized_hist, peaks, rel_height=0.5)

    for i, peak in enumerate(peaks):
        peak_height = properties['peak_heights'][i]
        peak_index = bin_centers_rounded[peak]
        peak_width = peak_widths_result[0][i]
        peak_data_list.append({
            'Array Index': idx,
            'Peak Value': peak_height,
            'Peak Degree (Index)': peak_index,
            'Peak Width': peak_width
        })

    ax.plot(bin_centers_rounded[peaks], normalized_hist[peaks], 'ro', label="Peaks")
    ax.legend()


for ax in axes[len(modified_fitted_osi_angle_deg_mod):]:
    ax.axis('off')

plt.tight_layout()
output_directory = folder_path
os.makedirs(output_directory, exist_ok=True)
output_image_path = os.path.join(output_directory, 'histogram_grid_shifted.png')
plt.savefig(output_image_path)
plt.show()



peak_data_df_normalised = pd.DataFrame(peak_data_list)
output_file = os.path.join(output_directory, 'normalized_peaks_data.xlsx')
peak_data_df_normalised.to_excel(output_file, index=False)
print(f"Normalized peaks data saved to: {output_file}")

example_idx = 1
print(f"Raw histogram for array {example_idx}:")
print(histogram_arrays[example_idx])
print("\n")

print(f"Normalized histogram for array {example_idx}:")
print(normalized_histogram_arrays[example_idx])
print("\n")

print("Peak Data for All Arrays:")
print(peak_data_df_normalised)


#HORIZONTAL PEAK RANGE RAW AND EDITED

min_peak_index_h = 50
max_peak_index_h = 100
filtered_peaks_df_h_normalised_noedit = peak_data_df_normalised[(peak_data_df_normalised['Peak Degree (Index)'] >= min_peak_index_h) & (peak_data_df_normalised['Peak Degree (Index)'] <= max_peak_index_h)]
print(filtered_peaks_df_h_normalised_noedit)

os.makedirs(output_directory, exist_ok=True)
csv_file_name = 'horizontal_peaks_normalised_noedit.csv'
csv_file_path = os.path.join(output_directory, csv_file_name)
filtered_peaks_df_h_normalised_noedit.to_csv(csv_file_path, index=False)
print(f"Data saved to: {csv_file_path}")

filtered_peaks_df_h_normalised = peak_data_df_normalised[
    (peak_data_df_normalised['Peak Degree (Index)'] >= min_peak_index_h) & 
    (peak_data_df_normalised['Peak Degree (Index)'] <= max_peak_index_h)
]


filtered_peaks_df_h_normalised = filtered_peaks_df_h_normalised.loc[filtered_peaks_df_h_normalised.groupby('Array Index')['Peak Value'].idxmax()]
existing_indices = filtered_peaks_df_h_normalised['Array Index'].unique()
all_indices = np.arange(existing_indices.min(), existing_indices.max() + 1)
missing_indices = [index for index in all_indices if index not in existing_indices]
new_rows = pd.DataFrame({
    'Array Index': missing_indices,
    'Peak Value': [0.000001] * len(missing_indices),
    'Peak Degree (Index)': [np.nan] * len(missing_indices),
    'Peak Width': [np.nan] * len(missing_indices)
})

if 1 not in existing_indices:
    new_rows = pd.concat([new_rows, pd.DataFrame({
        'Array Index': [1],
        'Peak Value': [0.000001],
        'Peak Degree (Index)': [np.nan],
        'Peak Width': [np.nan]
    })], ignore_index=True)

if 20 not in existing_indices:
    new_rows = pd.concat([new_rows, pd.DataFrame({
        'Array Index': [20],
        'Peak Value': [0.000001],
        'Peak Degree (Index)': [np.nan],
        'Peak Width': [np.nan]
    })], ignore_index=True)
    

filtered_peaks_df_h_normalised = pd.concat([filtered_peaks_df_h_normalised, new_rows], ignore_index=True)
filtered_peaks_df_h_normalised.sort_values(by='Array Index', inplace=True)
print(filtered_peaks_df_h_normalised)

os.makedirs(output_directory, exist_ok=True)
csv_file_name = 'horizontal_peaks_normalised.csv'
csv_file_path = os.path.join(output_directory, csv_file_name)
filtered_peaks_df_h_normalised.to_csv(csv_file_path, index=False)
print(f"Data saved to: {csv_file_path}")



#Vertical Peak Range
min_peak_index_v = 155
max_peak_index_v = 205

filtered_peaks_df_v_normalised_noedit = peak_data_df_normalised[
    (peak_data_df_normalised['Peak Degree (Index)'] >= min_peak_index_v) & 
    (peak_data_df_normalised['Peak Degree (Index)'] <= max_peak_index_v)
]
print(filtered_peaks_df_v_normalised_noedit)

os.makedirs(output_directory, exist_ok=True)
csv_file_name = 'vertical_peaks_normalised_noedit.csv'
csv_file_path = os.path.join(output_directory, csv_file_name)
filtered_peaks_df_v_normalised_noedit.to_csv(csv_file_path, index=False)
print(f"Data saved to: {csv_file_path}")

filtered_peaks_df_v_normalised = peak_data_df_normalised[
    (peak_data_df_normalised['Peak Degree (Index)'] >= min_peak_index_v) & 
    (peak_data_df_normalised['Peak Degree (Index)'] <= max_peak_index_v)
]
filtered_peaks_df_v_normalised = filtered_peaks_df_v_normalised.loc[
    filtered_peaks_df_v_normalised.groupby('Array Index')['Peak Value'].idxmax()
]
existing_indices = filtered_peaks_df_v_normalised['Array Index'].unique()
all_indices = np.arange(existing_indices.min(), existing_indices.max() + 1)
missing_indices = [index for index in all_indices if index not in existing_indices]

new_rows = pd.DataFrame({
    'Array Index': missing_indices,
    'Peak Value': [0.000001] * len(missing_indices),
    'Peak Degree (Index)': [np.nan] * len(missing_indices),
    'Peak Width': [np.nan] * len(missing_indices)
})

if 1 not in existing_indices:
    new_rows = pd.concat([new_rows, pd.DataFrame({
        'Array Index': [1],
        'Peak Value': [0.000001],
        'Peak Degree (Index)': [np.nan],
        'Peak Width': [np.nan]
    })], ignore_index=True)

if 20 not in existing_indices:
    new_rows = pd.concat([new_rows, pd.DataFrame({
        'Array Index': [20],
        'Peak Value': [0.000001],
        'Peak Degree (Index)': [np.nan],
        'Peak Width': [np.nan]
    })], ignore_index=True)


filtered_peaks_df_v_normalised = pd.concat([filtered_peaks_df_v_normalised, new_rows], ignore_index=True)
filtered_peaks_df_v_normalised.sort_values(by='Array Index', inplace=True)
print(filtered_peaks_df_v_normalised)
os.makedirs(output_directory, exist_ok=True)
csv_file_name = 'vertical_peaks_normalised.csv'
csv_file_path = os.path.join(output_directory, csv_file_name)
filtered_peaks_df_v_normalised.to_csv(csv_file_path, index=False)
print(f"Data saved to: {csv_file_path}")

# Getting a list of vertical indeces
column1 = filtered_peaks_df_h_normalised['Peak Value'].to_numpy()
column2 = filtered_peaks_df_v_normalised['Peak Value'].to_numpy()
result_v = column2 / (column2 + column1)
result_v_df = pd.DataFrame(result_v, columns=['Vertical Index Normalised'])
print(result_v_df)

os.makedirs(output_directory, exist_ok=True)
index_csv_file_name = 'vertical_index_normalised.csv'
index_csv_file_path = os.path.join(output_directory, index_csv_file_name)
result_v_df.to_csv(index_csv_file_path, index=False)
print(f"Data saved to: {index_csv_file_path}")

log_array_v = (np.log2((result_v_df)+0.5))
print(log_array_v)
type(log_array_v)

logindex_csv_file_name = 'vertical_index_normalised_log.csv'
logindex_csv_file_path = os.path.join(output_directory, logindex_csv_file_name)
log_array_v.to_csv(logindex_csv_file_path, index=False)
print(f"Data saved to: {logindex_csv_file_path}")

log_array_v['Vertical Index Normalised'] = log_array_v['Vertical Index Normalised'].astype(float)
sorted_data = log_array_v['Vertical Index Normalised'].sort_values()
sorted_values = sorted_data.values
sorted_indices = sorted_data.index + 1


plt.figure(figsize=(10, 6))
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.bar(range(1, len(sorted_values) + 1), sorted_values, 
        color=['palevioletred' if i >= 0 else 'blueviolet' for i in sorted_values], width=0.8)
plt.ylabel('Vertical Index Transformed', fontsize=20)
plt.ylim([-1, 1])
plt.yticks(np.arange(-1, 2, 1), fontsize = 18)

plt.xlabel('Individual Animal', fontsize=20)
plt.xticks([1], labels= (''))
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

save_folder = folder_path
os.makedirs(save_folder, exist_ok=True)
save_path_png = os.path.join(save_folder, 'data_in_relation_to_zero_aligned_with_indices.png')
save_path_svg = os.path.join(save_folder, 'data_in_relation_to_zero_aligned_with_indices.svg')
plt.savefig(save_path_png, dpi=600)
plt.savefig(save_path_svg, format='svg')
plt.savefig(save_folder)
plt.show()

print(f"Plot saved to: {save_folder}")
