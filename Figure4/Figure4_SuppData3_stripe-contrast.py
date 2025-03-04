########################################################################################################################################################################
# AVERAGE SPEED

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0_vertical_left_90': 0,
                '0.1_vertical_left_90': 1,
                '0.2_vertical_left_90': 2,
                '0.3_vertical_left_90': 3,
                '0.4_vertical_left_90': 4,
                '0.5_vertical_left_90': 5,
                '0.6_vertical_left_90': 6,
                '0.7_vertical_left_90': 7,
                '0.8_vertical_left_90': 8,
                '0.9_vertical_left_90': 9,
                '1_vertical_left_90': 10}
number_of_stims = len(stim_indexer)

def label_bouts(path):
    df = pd.read_hdf(path,key ="analysed")
    df = df[np.logical_and(df.index.get_level_values('window_time') > 10, df.index.get_level_values('window_time') < 20)]

    bout_angle_threshold = 5
    df['time'] = np.nan
    df['left_bouts'] = np.nan
    df['right_bouts'] = np.nan
    df['straight_bouts'] = np.nan
    df['bout_orientation'] = np.nan
    # bigger than | right
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "bout_orientation"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "right_bouts"] = 0
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "left_bouts"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "straight_bouts"] = 0
    # smaller | than left
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "bout_orientation"] = -1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "right_bouts"] = 1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "left_bouts"] = 0
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "straight_bouts"] = 0
    # absolute value | straight
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "bout_orientation"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "right_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "left_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "straight_bouts"] = 1
    return df


def group_df_by_trial_percentage_left(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).sum()
    df_gr['average_speed'] = np.nan
    df_gr['average_speed'] = df_gr['left_bouts']/(df_gr['left_bouts'] + df_gr['right_bouts'])
    return df_gr

def group_df_by_trial_average(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).mean()
    return df_gr


def get_mean_and_sem_stimuli(df):
    stimuli = df
    fishes = df.index.unique('folder_name')
    stimuli = df.index.unique('stimulus_name')
    list_of_means_per_stimulus = []
    print(stimuli)
    list_of_sems_per_stimulus = []
    for stimulus in stimuli:
        list_of_means_per_stim_per_fish = []
        for fish in fishes:
            df_stimulus_fish  = df.xs((fish, stimulus), level=('folder_name', 'stimulus_name'))
            if len(df_stimulus_fish)>0:
                list_of_means_per_stim_per_fish.append(np.nanmean(df_stimulus_fish['average_speed'].values))
        list_of_means_per_stimulus.append(np.nanmean(list_of_means_per_stim_per_fish))
        list_of_sems_per_stimulus.append(np.nanstd(list_of_means_per_stim_per_fish) / np.sqrt(len(list_of_means_per_stim_per_fish)))
    return list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli


def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot):

    plt.rcParams["figure.figsize"] = (8, 5)
    stimulus_names = df1.index.unique('stimulus_name')
    x = [stim_indexer[stimulus_names[i]] for i in range (len(stimulus_names))]
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, label=None, linewidths=2.5)
    plt.xticks(x,labels=[stimulus_names[i].split('_')[0] for i in range(len(stimulus_names))],rotation=0, fontsize=18)
    plt.yticks([0.25, 0.3, 0.35, 0.4, 0.45], labels=['1.5', '1.8', '2.1', '2.4', '2.7'],fontsize=18)
    plt.xlabel(xlabel="Stripe contrast", fontsize=20)
    plt.ylabel(ylabel="Speed (cm/s)", fontsize=20)
    plt.ylim(0.25, 0.45)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

path1 = Path(r'E:\3_behaviour\stripe-contrast\2024-no-gradient\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot = "salmon")
print("Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()


########################################################################################################################################################################
# BOUT DURATION

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0_vertical_left_90': 0,
                '0.1_vertical_left_90': 1,
                '0.2_vertical_left_90': 2,
                '0.3_vertical_left_90': 3,
                '0.4_vertical_left_90': 4,
                '0.5_vertical_left_90': 5,
                '0.6_vertical_left_90': 6,
                '0.7_vertical_left_90': 7,
                '0.8_vertical_left_90': 8,
                '0.9_vertical_left_90': 9,
                '1_vertical_left_90': 10}
number_of_stims = len(stim_indexer)

def label_bouts(path):
    df = pd.read_hdf(path,key ="analysed")
    df = df[np.logical_and(df.index.get_level_values('window_time') > 10, df.index.get_level_values('window_time') < 20)]

    bout_angle_threshold = 5
    df['time'] = np.nan
    df['left_bouts'] = np.nan
    df['right_bouts'] = np.nan
    df['straight_bouts'] = np.nan
    df['bout_orientation'] = np.nan
    # bigger than | right
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "bout_orientation"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "right_bouts"] = 0
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "left_bouts"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "straight_bouts"] = 0
    # smaller | than left
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "bout_orientation"] = -1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "right_bouts"] = 1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "left_bouts"] = 0
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "straight_bouts"] = 0
    # absolute value | straight
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "bout_orientation"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "right_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "left_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "straight_bouts"] = 1
    return df


def group_df_by_trial_percentage_left(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).sum()
    df_gr['duration'] = np.nan
    df_gr['duration'] = df_gr['left_bouts']/(df_gr['left_bouts'] + df_gr['right_bouts'])
    return df_gr

def group_df_by_trial_average(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).mean()
    return df_gr


def get_mean_and_sem_stimuli(df):
    stimuli = df
    fishes = df.index.unique('folder_name')
    stimuli = df.index.unique('stimulus_name')
    list_of_means_per_stimulus = []
    print(stimuli)
    list_of_sems_per_stimulus = []
    for stimulus in stimuli:
        list_of_means_per_stim_per_fish = []
        for fish in fishes:
            df_stimulus_fish  = df.xs((fish, stimulus), level=('folder_name', 'stimulus_name'))
            if len(df_stimulus_fish)>0:
                list_of_means_per_stim_per_fish.append(np.nanmean(df_stimulus_fish['duration'].values))
        list_of_means_per_stimulus.append(np.nanmean(list_of_means_per_stim_per_fish))
        list_of_sems_per_stimulus.append(np.nanstd(list_of_means_per_stim_per_fish) / np.sqrt(len(list_of_means_per_stim_per_fish)))
    return list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli


def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot):
    plt.rcParams["figure.figsize"] = (8, 5)
    stimulus_names = df1.index.unique('stimulus_name')
    x = [stim_indexer[stimulus_names[i]] for i in range (len(stimulus_names))]
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, label=None, linewidths=2.5)
    plt.xticks(x,labels=[stimulus_names[i].split('_')[0] for i in range(len(stimulus_names))],rotation=0, fontsize=18)
    plt.yticks([0.12, 0.13, 0.14, 0.15, 0.16, 0.17], labels=['120', '130', '140', '150', '160', '170'], fontsize=18)
    plt.ylim(0.12, 0.17)
    plt.xlabel(xlabel="Stripe contrast", fontsize=20)
    plt.ylabel(ylabel="Bout duration (msec)", fontsize=20)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

path1 = Path(r'E:\3_behaviour\stripe-contrast\2024-no-gradient\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot = "salmon")
print("Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()

########################################################################################################################################################################
# DISTANCE CHANGE

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0_vertical_left_90': 0,
                '0.1_vertical_left_90': 1,
                '0.2_vertical_left_90': 2,
                '0.3_vertical_left_90': 3,
                '0.4_vertical_left_90': 4,
                '0.5_vertical_left_90': 5,
                '0.6_vertical_left_90': 6,
                '0.7_vertical_left_90': 7,
                '0.8_vertical_left_90': 8,
                '0.9_vertical_left_90': 9,
                '1_vertical_left_90': 10}
number_of_stims = len(stim_indexer)

def label_bouts(path):
    df = pd.read_hdf(path,key ="analysed")
    df = df[np.logical_and(df.index.get_level_values('window_time') > 10, df.index.get_level_values('window_time') < 20)]

    bout_angle_threshold = 5
    df['time'] = np.nan
    df['left_bouts'] = np.nan
    df['right_bouts'] = np.nan
    df['straight_bouts'] = np.nan
    df['bout_orientation'] = np.nan
    # bigger than | right
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "bout_orientation"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "right_bouts"] = 0
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "left_bouts"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "straight_bouts"] = 0
    # smaller | than left
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "bout_orientation"] = -1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "right_bouts"] = 1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "left_bouts"] = 0
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "straight_bouts"] = 0
    # absolute value | straight
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "bout_orientation"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "right_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "left_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "straight_bouts"] = 1
    return df


def group_df_by_trial_percentage_left(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).sum()
    df_gr['distance_change'] = np.nan
    df_gr['distance_change'] = df_gr['left_bouts']/(df_gr['left_bouts'] + df_gr['right_bouts'])
    return df_gr

def group_df_by_trial_average(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).mean()
    return df_gr


def get_mean_and_sem_stimuli(df):
    stimuli = df
    fishes = df.index.unique('folder_name')
    stimuli = df.index.unique('stimulus_name')
    list_of_means_per_stimulus = []
    print(stimuli)
    list_of_sems_per_stimulus = []
    for stimulus in stimuli:
        list_of_means_per_stim_per_fish = []
        for fish in fishes:
            df_stimulus_fish  = df.xs((fish, stimulus), level=('folder_name', 'stimulus_name'))
            if len(df_stimulus_fish)>0:
                list_of_means_per_stim_per_fish.append(np.nanmean(df_stimulus_fish['distance_change'].values))
        list_of_means_per_stimulus.append(np.nanmean(list_of_means_per_stim_per_fish))
        list_of_sems_per_stimulus.append(np.nanstd(list_of_means_per_stim_per_fish) / np.sqrt(len(list_of_means_per_stim_per_fish)))
    return list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli


def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot):
    plt.rcParams["figure.figsize"] = (8, 5)
    stimulus_names = df1.index.unique('stimulus_name')
    x = [stim_indexer[stimulus_names[i]] for i in range (len(stimulus_names))]
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, label=None, linewidths=2.5)
    plt.xticks(x,labels=[stimulus_names[i].split('_')[0] for i in range(len(stimulus_names))],rotation=0, fontsize=18)
    plt.yticks([0.0333, 0.041667, 0.05, 0.05833, 0.066667], labels=['2.0', '2.5', '3.0', '3.5', '4.0'],fontsize=18)
    plt.ylim(0.0333, 0.066667)
    plt.xlabel(xlabel="Stripe contrast", fontsize=20)
    plt.ylabel(ylabel="Bout length (mm)", fontsize=20)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)


path1 = Path(r'E:\3_behaviour\stripe-contrast\2024-no-gradient\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot = "salmon")
print("Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()


########################################################################################################################################################################
# EVENT FREQUENCY HZ

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update({'font.size': 12})
stim_indexer = {'0_vertical_left_90': 0,
                '0.1_vertical_left_90': 1,
                '0.2_vertical_left_90': 2,
                '0.3_vertical_left_90': 3,
                '0.4_vertical_left_90': 4,
                '0.5_vertical_left_90': 5,
                '0.6_vertical_left_90': 6,
                '0.7_vertical_left_90': 7,
                '0.8_vertical_left_90': 8,
                '0.9_vertical_left_90': 9,
                '1_vertical_left_90': 10}
number_of_stims = len(stim_indexer)

def label_bouts(path):
    df = pd.read_hdf(path,key ="analysed")
    df = df[np.logical_and(df.index.get_level_values('window_time') > 10, df.index.get_level_values('window_time') < 20)]

    bout_angle_threshold = 5
    df['time'] = np.nan
    df['left_bouts'] = np.nan
    df['right_bouts'] = np.nan
    df['straight_bouts'] = np.nan
    df['bout_orientation'] = np.nan
    # bigger than | right
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "bout_orientation"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "right_bouts"] = 0
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "left_bouts"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "straight_bouts"] = 0
    # smaller | than left
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "bout_orientation"] = -1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "right_bouts"] = 1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "left_bouts"] = 0
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "straight_bouts"] = 0
    # absolute value | straight
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "bout_orientation"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "right_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "left_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "straight_bouts"] = 1
    return df


def group_df_by_trial_percentage_left(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).sum()
    df_gr['event_freq'] = np.nan
    df_gr['event_freq'] = df_gr['left_bouts']/(df_gr['left_bouts'] + df_gr['right_bouts'])
    return df_gr

def group_df_by_trial_average(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).mean()
    return df_gr


def get_mean_and_sem_stimuli(df):
    stimuli = df
    fishes = df.index.unique('folder_name')
    stimuli = df.index.unique('stimulus_name')
    list_of_means_per_stimulus = []
    print(stimuli)
    list_of_sems_per_stimulus = []
    for stimulus in stimuli:
        list_of_means_per_stim_per_fish = []
        for fish in fishes:
            df_stimulus_fish  = df.xs((fish, stimulus), level=('folder_name', 'stimulus_name'))
            if len(df_stimulus_fish)>0:
                list_of_means_per_stim_per_fish.append(np.nanmean(df_stimulus_fish['event_freq'].values))
        list_of_means_per_stimulus.append(np.nanmean(list_of_means_per_stim_per_fish))
        list_of_sems_per_stimulus.append(np.nanstd(list_of_means_per_stim_per_fish) / np.sqrt(len(list_of_means_per_stim_per_fish)))
    return list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli

def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot):
    plt.rcParams["figure.figsize"] = (8, 5)
    stimulus_names = df1.index.unique('stimulus_name')
    x = [stim_indexer[stimulus_names[i]] for i in range (len(stimulus_names))]
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, label=None, linewidths=2.5)
    plt.xticks(x,labels=[stimulus_names[i].split('_')[0] for i in range(len(stimulus_names))],rotation=0, fontsize=18)
    plt.yticks([20, 22, 24, 26, 28, 30], labels=['1', '1.1', '1.2', '1.3', '1.4', '1.5'], fontsize=18)
    plt.xlabel(xlabel="Stripe contrast", fontsize=20)
    plt.ylabel(ylabel="Swim rate (Hz)", fontsize=20)
    plt.ylim(20, 30)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

path1 = Path(r'E:\3_behaviour\stripe-contrast\2024-no-gradient\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot = "salmon")
print("Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()



########################################################################################################################################################################
# IBI

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update({'font.size': 12})
stim_indexer = {'0_vertical_left_90': 0,
                '0.1_vertical_left_90': 1,
                '0.2_vertical_left_90': 2,
                '0.3_vertical_left_90': 3,
                '0.4_vertical_left_90': 4,
                '0.5_vertical_left_90': 5,
                '0.6_vertical_left_90': 6,
                '0.7_vertical_left_90': 7,
                '0.8_vertical_left_90': 8,
                '0.9_vertical_left_90': 9,
                '1_vertical_left_90': 10}
number_of_stims = len(stim_indexer)

def label_bouts(path):
    df = pd.read_hdf(path,key ="analysed")
    df = df[np.logical_and(df.index.get_level_values('window_time') > 10, df.index.get_level_values('window_time') < 20)]

    bout_angle_threshold = 5
    df['time'] = np.nan
   # df1['time'] = df1['end_time']
    df['left_bouts'] = np.nan
    df['right_bouts'] = np.nan
    df['straight_bouts'] = np.nan
    df['bout_orientation'] = np.nan
    # bigger than | right
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "bout_orientation"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "right_bouts"] = 0
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "left_bouts"] = 1
    df.loc[df['estimated_orientation_change'] > bout_angle_threshold, "straight_bouts"] = 0
    # smaller | than left
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "bout_orientation"] = -1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "right_bouts"] = 1
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "left_bouts"] = 0
    df.loc[df['estimated_orientation_change'] < - bout_angle_threshold, "straight_bouts"] = 0
    # absolute value | straight
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "bout_orientation"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "right_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "left_bouts"] = 0
    df.loc[abs(df['estimated_orientation_change']) < bout_angle_threshold, "straight_bouts"] = 1
    return df


def group_df_by_trial_percentage_left(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).sum()
    df_gr['interbout_interval'] = np.nan
    df_gr['interbout_interval'] = df_gr['left_bouts']/(df_gr['left_bouts'] + df_gr['right_bouts'])
    return df_gr

def group_df_by_trial_average(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).mean()
    return df_gr


def get_mean_and_sem_stimuli(df):
    stimuli = df
    fishes = df.index.unique('folder_name')
    stimuli = df.index.unique('stimulus_name')
    list_of_means_per_stimulus = []
    print(stimuli)
    list_of_sems_per_stimulus = []
    for stimulus in stimuli:
        list_of_means_per_stim_per_fish = []
        for fish in fishes:
            df_stimulus_fish  = df.xs((fish, stimulus), level=('folder_name', 'stimulus_name'))
            if len(df_stimulus_fish)>0:
                list_of_means_per_stim_per_fish.append(np.nanmean(df_stimulus_fish['interbout_interval'].values))
        list_of_means_per_stimulus.append(np.nanmean(list_of_means_per_stim_per_fish))
        list_of_sems_per_stimulus.append(np.nanstd(list_of_means_per_stim_per_fish) / np.sqrt(len(list_of_means_per_stim_per_fish)))
    return list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli


def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot):
    plt.rcParams["figure.figsize"] = (8, 5)
    stimulus_names = df1.index.unique('stimulus_name')
    x = [stim_indexer[stimulus_names[i]] for i in range (len(stimulus_names))]
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, label=None, linewidths=2.5)
    plt.xticks(x,labels=[stimulus_names[i].split('_')[0] for i in range(len(stimulus_names))],rotation=0, fontsize=18)
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2], labels=['0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'],fontsize=18)
    plt.xlabel(xlabel="Stripe contrast", fontsize=20)
    plt.ylabel(ylabel="Interbout interval (s)", fontsize=20)
    plt.ylim(0.5, 1.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)


path1 = Path(r'E:\3_behaviour\stripe-contrast\2024-no-gradient\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,color_of_plot = "salmon")
print("Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")


plt.show()





########################################################################################################################################################################
# PERCENTAGE LEFT

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0_vertical_left_90': 0,
                '0.1_vertical_left_90': 1,
                '0.2_vertical_left_90': 2,
                '0.3_vertical_left_90': 3,
                '0.4_vertical_left_90': 4,
                '0.5_vertical_left_90': 5,
                '0.6_vertical_left_90': 6,
                '0.7_vertical_left_90': 7,
                '0.8_vertical_left_90': 8,
                '0.9_vertical_left_90': 9,
                '1_vertical_left_90': 10}
number_of_stims = len(stim_indexer)

def group_df_by_trial_average(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).mean()
    return df_gr


def get_mean_and_sem_stimuli(df):
    stimuli = df
    fishes = df.index.unique('folder_name')
    stimuli = df.index.unique('stimulus_name')
    list_of_means_per_stimulus = []
    print(stimuli)
    list_of_sems_per_stimulus = []
    for stimulus in stimuli:
        list_of_means_per_stim_per_fish = []
        for fish in fishes:
            df_stimulus_fish  = df.xs((fish, stimulus), level=('folder_name', 'stimulus_name'))
            if len(df_stimulus_fish)>0:
                list_of_means_per_stim_per_fish.append(np.nanmean(df_stimulus_fish['percentage_left'].values))
        list_of_means_per_stimulus.append(np.nanmean(list_of_means_per_stim_per_fish))
        list_of_sems_per_stimulus.append(np.nanstd(list_of_means_per_stim_per_fish) / np.sqrt(len(list_of_means_per_stim_per_fish)))
    return list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli


def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset,color_of_plot):
    plt.rcParams["figure.figsize"] = (8, 5)
    stimulus_names = df.index.unique('stimulus_name')
    x = [stim_indexer[stimulus_names[i]] for i in range (len(stimulus_names))]
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, label=None, linewidths=2.5)
    plt.xticks(x,labels=[stimulus_names[i].split('_')[0] for i in range(len(stimulus_names))],rotation=0, fontsize=18)
    plt.ylim(0.45,0.8)
    plt.yticks([0.5, 0.6, 0.7, 0.8], labels= ('0.5','0.6','0.7','0.8'),fontsize=18)
    plt.xlabel(xlabel= "Stripe contrast", fontsize=20)
    plt.ylabel(ylabel= "Proportion turn to parallel stimuli", fontsize=20)
    plt.axhline(0.5, color='dimgrey', linestyle="--", linewidth=2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

path1 = Path(r'E:\3_behaviour\stripe-contrast\2024-no-gradient\Analysis\data_analysed.hdf5')
df = pd.read_hdf(path1, key="analysed")
df = df[np.logical_and(df.index.get_level_values('window_time') > 10, df.index.get_level_values('window_time') < 20)]
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "control",color_of_plot = "salmon")
print("Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()
