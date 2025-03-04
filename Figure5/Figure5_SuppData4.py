######################################################################################################################
#AVERAGE SPEED

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0': 0,
                '30': 1,
                '60': 2,
                '90': 3,
                '120': 4,
                '150': 5}
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
    list_of_sems_per_stimulus = []
    for stimulus in stimuli:
        list_of_means_per_stim_per_fish = []
        for fish in fishes: # iterate through fish for each stimuli!
            df_stimulus_fish  = df.xs((fish, stimulus), level=('folder_name', 'stimulus_name'))
            if len(df_stimulus_fish)>0:
                list_of_means_per_stim_per_fish.append(np.nanmean(df_stimulus_fish['average_speed'].values))
        list_of_means_per_stimulus.append(np.nanmean(list_of_means_per_stim_per_fish))
        list_of_sems_per_stimulus.append(np.nanstd(list_of_means_per_stim_per_fish) / np.sqrt(len(list_of_means_per_stim_per_fish)))
    return list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli

def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,color_of_plot):
    plt.rcParams["figure.figsize"] = (7, 5)
    x = np.array([0, 4, 5, 1, 2, 3])
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot, label=None)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, linewidth=2.5, label=None)
    plt.xticks(x,labels=['0', '120', '150', '30', '60', '90'],rotation=0, fontsize=18)
    plt.yticks([0.25, 0.3, 0.35, 0.4], labels=['1.5', '1.8', '2.1', '2.4'], fontsize=18)
    plt.ylim(0.25, 0.4)
    plt.xlabel(xlabel= "Degrees in opposing eye", fontsize=20)
    plt.ylabel(ylabel= "Average speed (cm/s)", fontsize=20)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

path1 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\control\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "control",color_of_plot = "gold")
print("Control Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path2 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\vertical\Analysis\data_analysed.hdf5')
df2 = label_bouts(path2)
df2 = group_df_by_trial_average(df2)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df2)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "vertical",color_of_plot = "dodgerblue")
print("Vertical Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path3 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\horizontal\Analysis\data_analysed.hdf5')
df3 = label_bouts(path3)
df3 = group_df_by_trial_average(df3)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df3)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "horizontal" ,color_of_plot = "deeppink")
print("Horizontal Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()



######################################################################################################################
#BOUT DURATION

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0': 0,
                '30': 1,
                '60': 2,
                '90': 3,
                '120': 4,
                '150': 5}
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

def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,color_of_plot):
    plt.rcParams["figure.figsize"] = (7, 5)
    x = np.array([0, 4, 5, 1, 2, 3])
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot, label=None)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, linewidth=2.5, label=None)
    plt.xticks(x,labels=['0', '120', '150', '30', '60', '90'],rotation=0, fontsize=18)
    plt.yticks([0.1, 0.12, 0.14, 0.16, 0.18, 0.2], labels=['100', '120', '140', '160', '180', '200'], fontsize=18)
    plt.ylim(0.1, 0.2)
    plt.xlabel(xlabel= "Degrees in opposing eye", fontsize=20)
    plt.ylabel(ylabel= "Bout duration (msec)", fontsize=20)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

path1 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\horizontal\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "control",color_of_plot = "deeppink")
print("Control Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path2 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\vertical\Analysis\data_analysed.hdf5')
df2 = label_bouts(path2)
df2 = group_df_by_trial_average(df2)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df2)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "vertical",color_of_plot = "dodgerblue")
print("Vertical Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path3 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\control\Analysis\data_analysed.hdf5')
df3 = label_bouts(path3)
df3 = group_df_by_trial_average(df3)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df3)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "horizontal" ,color_of_plot = "gold")
print("Horizontal Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")


plt.show()


######################################################################################################################
#SWIM FREQUENCY (HZ)

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0': 0,
                '30': 1,
                '60': 2,
                '90': 3,
                '120': 4,
                '150': 5}
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

def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,color_of_plot):
    plt.rcParams["figure.figsize"] = (7, 5) #15, 7
    x = np.array([0, 4, 5, 1, 2, 3])
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot, label=None)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, linewidth=2.5, label=None)
    plt.xticks(x,labels=['0', '120', '150', '30', '60', '90'],rotation=0, fontsize=18)
    plt.yticks([14, 16, 18, 20, 22, 24], labels=['0.7', '0.8', '0.9', '1.0', '1.1', '1.2'], fontsize=18)
    plt.ylim(14,24)
    plt.xlabel(xlabel= "Degrees in opposing eye", fontsize=20)
    plt.ylabel(ylabel= "Swim rate (Hz)", fontsize=20)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

path1 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\control\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "control",color_of_plot = "gold")
print("Control Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path2 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\vertical\Analysis\data_analysed.hdf5')
df2 = label_bouts(path2)
df2 = group_df_by_trial_average(df2)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df2)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "vertical",color_of_plot = "dodgerblue")
print("Vertical Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path3 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\horizontal\Analysis\data_analysed.hdf5')
df3 = label_bouts(path3)
df3 = group_df_by_trial_average(df3)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df3)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "horizontal" ,color_of_plot = "deeppink")
print("Horizontal Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()


######################################################################################################################
#DISTANCE CHANGE

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0': 0,
                '30': 1,
                '60': 2,
                '90': 3,
                '120': 4,
                '150': 5}
number_of_stims = len(stim_indexer)


# define bouts for all dataframes
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
    list_of_sems_per_stimulus = []
    for stimulus in stimuli: # iterate through stimuli
        list_of_means_per_stim_per_fish = [] # this list is renewed for each stimulus
        for fish in fishes: # iterate through fish for each stimuli!
            df_stimulus_fish  = df.xs((fish, stimulus), level=('folder_name', 'stimulus_name'))
            if len(df_stimulus_fish)>0:
                list_of_means_per_stim_per_fish.append(np.nanmean(df_stimulus_fish['distance_change'].values))
        list_of_means_per_stimulus.append(np.nanmean(list_of_means_per_stim_per_fish))
        list_of_sems_per_stimulus.append(np.nanstd(list_of_means_per_stim_per_fish) / np.sqrt(len(list_of_means_per_stim_per_fish)))
    return list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli


def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset,color_of_plot):
    plt.rcParams["figure.figsize"] = (7, 5) #15, 7
    x = np.array([0, 4, 5, 1, 2, 3])
    y = list_of_means_per_stimulus  # Effectively y = x**2
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot, label=None)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, linewidth=2.5, label=None)
    plt.xticks(x,labels=['0', '120', '150', '30', '60', '90'],rotation=0, fontsize=18)
    plt.yticks([0.0333, 0.041667, 0.05, 0.05833, 0.066667], labels=['2.0','2.5', '3.0', '3.5', '4.0'], fontsize=18)
    plt.ylim(0.0333, 0.066667)
    plt.xlabel(xlabel= "Degrees in opposing eye", fontsize=20)
    plt.ylabel(ylabel= "Bout length (mm)", fontsize=20)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)


path1 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\control\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "control",color_of_plot = "gold")
print("Control Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path2 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\vertical\Analysis\data_analysed.hdf5')
df2 = label_bouts(path2)
df2 = group_df_by_trial_average(df2)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df2)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "vertical",color_of_plot = "dodgerblue")
print("Vertical Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path3 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\horizontal\Analysis\data_analysed.hdf5')
df3 = label_bouts(path3)
df3 = group_df_by_trial_average(df3)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df3)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "horizontal" ,color_of_plot = "deeppink")
print("Horizontal Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()


######################################################################################################################
#IBI

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0': 0,
                '30': 1,
                '60': 2,
                '90': 3,
                '120': 4,
                '150': 5}
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


def plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset,color_of_plot):
    plt.rcParams["figure.figsize"] = (7, 5) #15, 7
    x = np.array([0, 4, 5, 1, 2, 3])
    y = list_of_means_per_stimulus  # Effectively y = x**2
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot, label=None)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, linewidth=2.5, label=None)
    plt.xticks(x,labels=['0', '120', '150', '30', '60', '90'],rotation=0, fontsize=18)
    plt.yticks([0.6, 0.8, 1.0, 1.2, 1.4], fontsize=18)
    plt.ylim(0.6,1.5)
    plt.xlabel(xlabel= "Degrees in opposing eye", fontsize=20)
    plt.ylabel(ylabel= "Interbout interval (s)", fontsize=20)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)


path1 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\control\Analysis\data_analysed.hdf5')
df1 = label_bouts(path1)
df1 = group_df_by_trial_average(df1)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "control",color_of_plot = "gold")
print("Control Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path2 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\vertical\Analysis\data_analysed.hdf5')
df2 = label_bouts(path2)
df2 = group_df_by_trial_average(df2)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df2)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "vertical",color_of_plot = "dodgerblue")
print("Vertical Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path3 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\horizontal\Analysis\data_analysed.hdf5')
df3 = label_bouts(path3)
df3 = group_df_by_trial_average(df3)
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df3)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "horizontal" ,color_of_plot = "deeppink")
print("Horizontal Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()


######################################################################################################################
#PERCENTAGE LEFT

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd


mpl.rcParams.update({'font.size': 12})

stim_indexer = {'0': 0,
                '30': 1,
                '60': 2,
                '90': 3,
                '120': 4,
                '150': 5}
number_of_stims = len(stim_indexer)


def group_df_by_trial_average(df):
    df_gr =df.groupby(["folder_name","stimulus_name"]).mean()
    return df_gr

def get_mean_and_sem_stimuli(df):
    stimuli = df
    fishes = df.index.unique('folder_name')
    stimuli = df.index.unique('stimulus_name')
    list_of_means_per_stimulus = []
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
    plt.rcParams["figure.figsize"] = (7, 5)
    x = np.array([0, 4, 5, 1, 2, 3])
    y = list_of_means_per_stimulus
    e = list_of_sems_per_stimulus
    plt.errorbar(x, y, yerr=e, ls="None", elinewidth=2.5, color=color_of_plot, label=None)
    plt.scatter(x, y, marker="o", color=color_of_plot, s=100, linewidths = 2.5, label=None)
    plt.xticks(x,labels=['0', '120', '150', '30', '60', '90'],rotation=0, fontsize=18)
    plt.ylim(0.45,0.7)
    plt.yticks([0.5, 0.6, 0.7], fontsize=18)
    plt.xlabel(xlabel= "Degrees in opposing eye", fontsize=20)
    plt.ylabel(ylabel= "Proportion turn to parallel stimuli", fontsize=20)
    plt.axhline(0.5, color='dimgrey', linestyle="--", linewidth=2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)


    for i in range(int((len(x) + len(y)) / 2)):
        sub_x = x[i:i + 3]
        sub_y = y[i:i + 3]
        model = np.poly1d(np.polyfit(sub_x, sub_y, 2))
        polyline = np.linspace(min(sub_x), max(sub_x), 6)
    x_new = np.linspace(min(x), max(x), 6)
    f = interp1d(x, y, kind='quadratic')
    plt.plot(x_new, f(x_new), color= color_of_plot, linestyle='dotted')
    print(x, y)


path1 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\control\Analysis\data_analysed.hdf5')
df1 = pd.read_hdf(path1, key="analysed")
df1 = df1[np.logical_and(df1.index.get_level_values('window_time') > 10, df1.index.get_level_values('window_time') < 20)]
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df1)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "control",color_of_plot = "gold")
print("Control Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path2 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\vertical\Analysis\data_analysed.hdf5')
df2 = pd.read_hdf(path2, key="analysed")
df2 = df2[np.logical_and(df2.index.get_level_values('window_time') > 10, df2.index.get_level_values('window_time') < 20)]
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df2)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "vertical",color_of_plot = "dodgerblue")
print("Vertical Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

path3 = Path(r'E:\3_behaviour\plastic-responses\no-gradient\horizontal\Analysis\data_analysed.hdf5')
df3 = pd.read_hdf(path3, key="analysed")
df3 = df3[np.logical_and(df3.index.get_level_values('window_time') > 10, df3.index.get_level_values('window_time') < 20)]
list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli = get_mean_and_sem_stimuli(df3)
plotting_func(list_of_means_per_stimulus,list_of_sems_per_stimulus,stimuli,name_of_dataset = "horizontal" ,color_of_plot = "deeppink")
print("Horizontal Data Means and SEMs:")
for stim, mean, sem in zip(stimuli, list_of_means_per_stimulus, list_of_sems_per_stimulus):
    print(f"Stimulus: {stim}, Mean: {mean}, SEM: {sem}")

plt.show()
