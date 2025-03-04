from scipy.io import loadmat
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

file_paths = {
    'control': 'path1',
    'vertical': 'path2',
    'horizontal': 'path3'
}

fitted_osi_angle_rad = {}
fitted_osi_angle_deg = {}
fitted_osi_angle_deg_mod = {}
mirrored_deg = {}
save_dir = 'path-to-save'
output_directory=save_dir


for label, path in file_paths.items():
    data = loadmat(path)
    variable = data['DEFAULTS_POST_GROUP']
    fitted_osi_angle_rad[label] = variable['fitted_osi_angle'].item()
    fitted_osi_angle_deg[label] = np.rad2deg(fitted_osi_angle_rad[label])
    mirrored_deg[label] = np.where(fitted_osi_angle_deg[label] > 180, 
                                   fitted_osi_angle_deg[label] - 180, 
                                   fitted_osi_angle_deg[label])

for label, angles in mirrored_deg.items():
    mask = (angles >= 0) & (angles <= 25)
    modified_angles = np.where(mask, angles + 180, angles)
    fitted_osi_angle_deg_mod[label] = modified_angles

fitted_osi_angle_rad_control = fitted_osi_angle_rad['control']
fitted_osi_angle_deg_control = fitted_osi_angle_deg['control']
fitted_osi_angle_deg_mod_control = fitted_osi_angle_deg_mod['control']

fitted_osi_angle_rad_vertical = fitted_osi_angle_rad['vertical']
fitted_osi_angle_deg_vertical = fitted_osi_angle_deg['vertical']
fitted_osi_angle_deg_mod_vertical = fitted_osi_angle_deg_mod['vertical']

fitted_osi_angle_rad_horizontal = fitted_osi_angle_rad['horizontal']
fitted_osi_angle_deg_horizontal = fitted_osi_angle_deg['horizontal']
fitted_osi_angle_deg_mod_horizontal = fitted_osi_angle_deg_mod['horizontal']



datasets = [fitted_osi_angle_deg_mod_control, fitted_osi_angle_deg_mod_vertical, fitted_osi_angle_deg_mod_horizontal]
dataset_labels = ['Control Environment', 'Vertical Environment', 'Horizontal Environment']
colors = ['gold', 'dodgerblue', 'deeppink']
plt.figure(figsize=(12, 8))

peak_data = []
for i, data in enumerate(datasets):
    hist, bin_edges = np.histogram(data, bins=36, range=(25, 205))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, _ = find_peaks(hist, distance=9)
    for peak in peaks:
        peak_degree = bin_centers[peak]
        peak_value = hist[peak]
        peak_data.append({
            'Dataset': dataset_labels[i], 
            'Peak Degree': peak_degree,
            'Frequency': peak_value
        })   

    plt.plot(bin_centers, hist, label=f'{dataset_labels[i]}', color=colors[i], linewidth = 2)
    plt.plot(bin_centers[peaks], hist[peaks], "x", color=colors[i], markersize=15, markeredgewidth=3, label=f'{dataset_labels[i]} Peaks')

plt.xlabel('Angle (Degrees)', fontsize = 22)
plt.ylabel('Number of Voxels', fontsize = 25)
plt.grid(False)
plt.xlim([25, 205])
plt.xticks(np.arange(30, 210, 30), fontsize = 20)
plt.yticks(fontsize = 16)
y_min, y_max = plt.ylim()
plt.rcParams['legend.fontsize'] = 16
plt.gca().add_patch(plt.Rectangle((50, 0), 50, max(hist), edgecolor='deeppink', linestyle='--', fill=False))
plt.gca().add_patch(plt.Rectangle((155, y_min), 40, y_max - y_min, edgecolor='dodgerblue', linestyle='--', fill=False))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['left'].set_color('darkgrey')
plt.gca().spines['bottom'].set_color('darkgrey')
plt.legend()

if not os.path.exists(save_dir):
    print(f"Error: The directory {save_dir} does not exist.")
else:
    save_path_png = os.path.join(save_dir, 'overlayed_histograms_with_peaks.png')
    save_path_svg = os.path.join(save_dir, 'overlayed_histograms_with_peaks.svg')
    plt.savefig(save_path_png)
    plt.savefig(save_path_svg)
    print(f"Plot saved as PNG to {save_path_png}")
    print(f"Plot saved as SVG to {save_path_svg}")
    plt.show()
plt.close()
df_peak_data = pd.DataFrame(peak_data)
excel_path = os.path.join(save_dir, 'peak_data.xlsx')
df_peak_data.to_excel(excel_path, index=False)
print(f"Peak data saved to {excel_path}")


df = pd.DataFrame(df_peak_data)
filtered_df_1 = df[(df['Dataset'] == 'Control Environment') & (df['Peak Degree'] >= 65) & (df['Peak Degree'] <= 120)]
control_horiz_peak_value = filtered_df_1['Frequency']
print("Horizontal Peak Value for Control Environment:")
print(control_horiz_peak_value.values)
filtered_df_2 = df[(df['Dataset'] == 'Control Environment') & (df['Peak Degree'] >= 160) & (df['Peak Degree'] <= 190)]
control_vert_peak_value = filtered_df_2['Frequency']
print("Vertical Peak Value for Control Environment:")
print(control_vert_peak_value.values)

#Vertical Indeces
filtered_df_3 = df[(df['Dataset'] == 'Vertical Environment') & (df['Peak Degree'] >= 65) & (df['Peak Degree'] <= 120)]
vertical_horiz_peak_value = filtered_df_3['Frequency']
print("Horizontal Peak Value for Vertical Environment:")
print(vertical_horiz_peak_value.values)
filtered_df_4 = df[(df['Dataset'] == 'Vertical Environment') & (df['Peak Degree'] >= 160) & (df['Peak Degree'] <= 190)]
vertical_vert_peak_value = filtered_df_4['Frequency']
print("Vertical Peak Value for Vertical Environment:")
print(vertical_vert_peak_value.values)

#Horizontal Indeces
filtered_df_5 = df[(df['Dataset'] == 'Horizontal Environment') & (df['Peak Degree'] >= 65) & (df['Peak Degree'] <= 120)]
horizontal_horiz_peak_value = filtered_df_5['Frequency']
print("Horizontal Peak Value for Horizontal Environment:")
print(horizontal_horiz_peak_value.values)
filtered_df_6 = df[(df['Dataset'] == 'Horizontal Environment') & (df['Peak Degree'] >= 160) & (df['Peak Degree'] <= 190)]
horizontal_vert_peak_value = filtered_df_6['Frequency']
print("Vertical Peak Value for Horizontal Environment:")
print(horizontal_vert_peak_value.values)


#control vertical index
control_index = (control_vert_peak_value.values)/((control_vert_peak_value.values)+(control_horiz_peak_value.values))
control_index_array = pd.Series(control_index)
print("Control Channels Vertical Angle Preference Index:")
print(control_index_array.values)

#vertical vertical index
vertical_index = vertical_vert_peak_value.values/(vertical_vert_peak_value.values+vertical_horiz_peak_value.values)
vertical_index_array = pd.Series(vertical_index)
print("Vertical Channels Vertical Angle Preference Index:")
print(vertical_index_array.values)

#horizontal vertical index
horizontal_index = horizontal_vert_peak_value.values/(horizontal_vert_peak_value.values+horizontal_horiz_peak_value.values)
horizontal_index_array = pd.Series(horizontal_index)
print("Horizontal Channels Vertical Angle Preference Index:")
print(horizontal_index_array.values)


groups = ['C', 'V', 'H']
horizontal_osi_counts = []
vertical_osi_counts = []
population_vertical_indices = []
control_horiz_peak_index = filtered_df_1['Peak Degree']
single_value_C_horiz_peak = control_horiz_peak_index.iloc[0]  
min_val_CH = single_value_C_horiz_peak - 10
max_val_CH = single_value_C_horiz_peak + 10
num_values_in_range_CH = np.sum((fitted_osi_angle_deg_mod_control >= min_val_CH) & (fitted_osi_angle_deg_mod_control <= max_val_CH))
print(f"H OSI for control group: {num_values_in_range_CH}")

# Vertical OSI for control group
control_vert_peak_index = filtered_df_2['Peak Degree']
single_value_C_vert_peak = control_vert_peak_index.iloc[0]  
min_val_CV = single_value_C_vert_peak - 10
max_val_CV = single_value_C_vert_peak + 10
num_values_in_range_CV = np.sum((fitted_osi_angle_deg_mod_control >= min_val_CV) & (fitted_osi_angle_deg_mod_control <= max_val_CV))
print(f"V OSI for control group: {num_values_in_range_CV}")

# Control population vertical index
control_population_vertical_index = num_values_in_range_CV / (num_values_in_range_CV + num_values_in_range_CH)
print(f"Control population V index: {control_population_vertical_index}")
horizontal_osi_counts.append(num_values_in_range_CH)
vertical_osi_counts.append(num_values_in_range_CV)
population_vertical_indices.append(control_population_vertical_index)


#Vertical group calculations
#Horizontal OSI for vertical group
vertical_horiz_peak_index = filtered_df_3['Peak Degree']
single_value_V_horiz_peak = vertical_horiz_peak_index.iloc[0]  
min_val_VH = single_value_V_horiz_peak - 10
max_val_VH = single_value_V_horiz_peak + 10
num_values_in_range_VH = np.sum((fitted_osi_angle_deg_mod_vertical >= min_val_VH) & (fitted_osi_angle_deg_mod_vertical <= max_val_VH))
print(f"H OSI for vertical group: {num_values_in_range_VH}")

#Vertical OSI for vertical group
vertical_vert_peak_index = filtered_df_4['Peak Degree']
single_value_V_vert_peak = vertical_vert_peak_index.iloc[0]  
min_val_VV = single_value_V_vert_peak - 10
max_val_VV = single_value_V_vert_peak + 10
num_values_in_range_VV = np.sum((fitted_osi_angle_deg_mod_vertical >= min_val_VV) & (fitted_osi_angle_deg_mod_vertical <= max_val_VV))
print(f"V OSI for vertical group: {num_values_in_range_VV}")
vertical_population_vertical_index = num_values_in_range_VV / (num_values_in_range_VV + num_values_in_range_VH)
print(f"Vertical population V index: {vertical_population_vertical_index}")

horizontal_osi_counts.append(num_values_in_range_VH)
vertical_osi_counts.append(num_values_in_range_VV)
population_vertical_indices.append(vertical_population_vertical_index)


#Horizontal group calculations
#Horizontal OSI for horizontal group
horizontal_horiz_peak_index = filtered_df_5['Peak Degree']
single_value_H_horiz_peak = horizontal_horiz_peak_index.iloc[0]  
min_val_HH = single_value_H_horiz_peak - 10
max_val_HH = single_value_H_horiz_peak + 10
num_values_in_range_HH = np.sum((fitted_osi_angle_deg_mod_horizontal >= min_val_HH) & (fitted_osi_angle_deg_mod_horizontal <= max_val_HH))
print(f"H OSI for horizontal group: {num_values_in_range_HH}")

# Vertical OSI for horizontal group
horizontal_vert_peak_index = filtered_df_6['Peak Degree']
single_value_H_vert_peak = horizontal_vert_peak_index.iloc[0]  
min_val_HV = single_value_H_vert_peak - 10
max_val_HV = single_value_H_vert_peak + 10
num_values_in_range_HV = np.sum((fitted_osi_angle_deg_mod_horizontal >= min_val_HV) & (fitted_osi_angle_deg_mod_horizontal <= max_val_HV))
print(f"V OSI for horizontal group: {num_values_in_range_HV}")

# Horizontal population vertical index
horizontal_population_vertical_index = num_values_in_range_HV / (num_values_in_range_HV + num_values_in_range_HH)
print(f"Horizontal population V index: {horizontal_population_vertical_index}")
horizontal_osi_counts.append(num_values_in_range_HH)
vertical_osi_counts.append(num_values_in_range_HV)
population_vertical_indices.append(horizontal_population_vertical_index)


total_counts = np.array(horizontal_osi_counts) + np.array(vertical_osi_counts)
horizontal_percentage = np.array(horizontal_osi_counts) / total_counts * 100
vertical_percentage = np.array(vertical_osi_counts) / total_counts * 100
os.makedirs(output_directory, exist_ok=True)

# Plot 1: Horizontal vs Vertical OSI as Stacked Bar Graph
fig, ax1 = plt.subplots(figsize=(7, 6))
x = np.arange(len(groups))
width = 1.0
ax1.bar(x, horizontal_percentage, width, color='blueviolet', edgecolor='black', alpha=0.7)
ax1.bar(x, vertical_percentage, width, bottom=horizontal_percentage, color='palevioletred', edgecolor='black', alpha=0.7)
ax1.yaxis.set_visible(False)
ax1.grid(False)
ax1.legend().set_visible(False)
ax1.set_xticks(x)
ax1.set_xticklabels(groups, fontweight='bold', fontsize = 22)

for i in range(len(groups)):
    ax1.text(i, horizontal_percentage[i] / 2, f'{horizontal_percentage[i]:.1f}%', 
             ha='center', va='center', color='white', fontweight='bold', fontsize = 22)
    ax1.text(i, horizontal_percentage[i] + vertical_percentage[i] / 2, 
             f'{vertical_percentage[i]:.1f}%', 
             ha='center', va='center', color='white', fontweight='bold', fontsize = 22)

legend_labels = ['Horizontal Preference','Vertical Preference']
legend_colors = ['palevioletred', 'blueviolet']
legend = ax1.legend(legend_labels, loc='upper right', bbox_to_anchor=(1, 1), framealpha=0.5)
frame = legend.get_frame()
frame.set_alpha(0.9)

output_image_path1_png = os.path.join(output_directory, 'HvsV_percentages_bar_graph.png')
output_image_path1_svg = os.path.join(output_directory, 'HvsV_percentages_bar_graph.svg')
plt.savefig(output_image_path1_png, dpi=600)
plt.savefig(output_image_path1_svg, format='svg')
plt.show()
plt.close(fig)
print(f"OSI percentages bar graph saved as PNG to: {output_image_path1_png}")
print(f"OSI percentages bar graph saved as SVG to: {output_image_path1_svg}")


#Plot 2: Population Vertical Index as Table
fig, ax2 = plt.subplots(figsize=(6, 4))
ax2.axis('off')

table_data = [['C', f'{population_vertical_indices[0]:.2f}'],
              ['V', f'{population_vertical_indices[1]:.2f}'],
              ['H', f'{population_vertical_indices[2]:.2f}']]
table = ax2.table(cellText=table_data, colLabels=['Group', 'Population Vertical Index'], loc='center', cellLoc='center')
row_colors = ['gold', 'dodgerblue', 'deeppink']
for i, color in enumerate(row_colors):
    table[(i + 1, 0)].set_facecolor(color)
    table[(i + 1, 1)].set_facecolor(color)
table.auto_set_font_size(False)
table.set_fontsize(15)
table.scale(1.5, 2)

output_image_path2_png = os.path.join(output_directory, 'population_vertical_index_table.png')
output_image_path2_svg = os.path.join(output_directory, 'population_vertical_index_table.svg')
plt.savefig(output_image_path2_png, dpi=600)
plt.savefig(output_image_path2_svg, format='svg')
plt.show()
plt.close(fig)
print(f"Population Vertical Index table saved as PNG to: {output_image_path2_png}")
print(f"Population Vertical Index table saved as SVG to: {output_image_path2_svg}")


#Plot 3: Population Vertical Index as Bar Graph
fig, ax3 = plt.subplots(figsize=(8, 6))
indices = np.arange(len(groups))
ax3.bar(indices, population_vertical_indices, color=['gold', 'dodgerblue', 'deeppink'], edgecolor='black')
ax3.set_yticks([0, 0.5, 1], labels= ('0','0.5','1'),fontsize=18)
ax3.set_xticks(indices)
ax3.set_xticklabels(groups, fontweight='bold', fontsize = 20)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(False)
ax3.spines['left'].set_linewidth(2)
ax3.spines['bottom'].set_linewidth(2)
ax3.spines['left'].set_color('dimgrey')
ax3.spines['bottom'].set_color('dimgrey')

for i, value in enumerate(population_vertical_indices):
    ax3.text(i, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize = 20)
output_image_path3_png = os.path.join(output_directory, 'population_vertical_index_bar_graph.png')
output_image_path3_svg = os.path.join(output_directory, 'population_vertical_index_bar_graph.svg')
plt.savefig(output_image_path3_png, dpi=600)
plt.savefig(output_image_path3_svg, format='svg')
plt.show()
plt.close(fig)
print(f"Population Vertical Index bar graph saved as PNG to: {output_image_path3_png}")
print(f"Population Vertical Index bar graph saved as SVG to: {output_image_path3_svg}")

#generate 3 histograms like the ones from matlab; looking the peaks
datasets = {
    'Control Environment': fitted_osi_angle_deg_mod_control,
    'Vertical Environment': fitted_osi_angle_deg_mod_vertical,
    'Horizontal Environment': fitted_osi_angle_deg_mod_horizontal
}
colors = ['black', 'deeppink', 'darkcyan']


for i, (label, data) in enumerate(datasets.items()):
    n_data = len(data)
    n_bins = int(np.sqrt(n_data))
    hist, bin_edges = np.histogram(data, bins=60)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, _ = find_peaks(hist, distance=25)
    results_half = peak_widths(hist, peaks, rel_height=0.5)
    widths = results_half[0] * np.diff(bin_edges).mean()

    for j, peak in enumerate(peaks):
        print(f"{label} - Peak {j+1}: Index = {peak}, Value = {hist[peak]}, Width = {widths[j]} at half maximum")
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, hist, width=np.diff(bin_edges).mean(), color=colors[i], alpha=0.7, label=f'{label} Histogram')
    plt.plot(bin_centers[peaks], hist[peaks], "x", color=colors[i])
    for j, peak in enumerate(peaks):
        plt.text(bin_centers[peak], hist[peak] + 0.05 * hist[peak], f'{hist[peak]}', 
                 ha='center', va='bottom', color=colors[i])
    plt.hlines(*results_half[1:], color=colors[i], linestyle='--', label=f'{label} Peak Widths')
    plt.xlabel('Angle')
    plt.ylabel('Number of Voxels')
    plt.title(f'Histogram with Detected Peaks and Widths - {label}')
    plt.xlim(25, 205)
    plt.ylim(0, 5000)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'histogram_{label.lower()}.png'))
    plt.show()
