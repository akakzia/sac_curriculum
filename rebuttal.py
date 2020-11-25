import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind
import pandas as pd


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


N_SAMPLES = 100
N_GOALS = 35

with open('rebuttal_semantic_rew.pkl', 'rb') as f:
    data_decstr = pkl.load(f)

with open('rebuttal_continuous_2.pkl', 'rb') as f:
    data_baseline = pkl.load(f)

# # Check if paired sampling
# for i in range(N_SAMPLES):
#     for j in range(N_GOALS):
#         assert str(data_decstr[i][j]['obs'][0][:55]) == str(data_baseline[i][j]['obs'][0][:55])

TIMES_SEMANTIC = [[] for _ in range(N_GOALS)]
TIMES_CONTINUOUS = [[] for _ in range(N_GOALS)]

DISTANCES_SEMANTIC = [[] for _ in range(N_GOALS)]
DISTANCES_CONTINUOUS = [[] for _ in range(N_GOALS)]

for i in range(N_SAMPLES):
    for j in range(N_GOALS):
        # Only take into account successful episodes
        if str(data_decstr[i][j]['ag_binary'][-1]) == str(data_decstr[i][j]['g_binary'][0]):
            try:
                time_success_decstr = np.where(data_decstr[i][j]['rewards'] == 1.)[0][0]
                TIMES_SEMANTIC[j].append(time_success_decstr)
                distance = 0
                for k in range(3):
                    distance += np.linalg.norm(data_decstr[i][j]['obs'][0][10+15*k:13+15*k] - data_decstr[i][j]['obs'][-1][10+15*k:13+15*k])
                DISTANCES_SEMANTIC[j].append(distance)
            except:
                pass
        # Same for baseline
        if data_baseline[i][j]['rewards'][-1] == 3.:
            time_success_continuous = np.where(data_baseline[i][j]['rewards'] == 3.)[0][0]
            TIMES_CONTINUOUS[j].append(time_success_continuous)
            distance = 0
            for k in range(3):
                distance += np.linalg.norm(
                    data_baseline[i][j]['obs'][0][10 + 15 * k:13 + 15 * k] - data_baseline[i][j]['obs'][-1][10 + 15 * k:13 + 15 * k])
            DISTANCES_CONTINUOUS[j].append(distance)

results = []
delta_times = []
max_p_value_times = -1

for i in range(N_GOALS):
    l = min(len(TIMES_SEMANTIC[i]), len(TIMES_CONTINUOUS[i]))
    results.append(ttest_rel(TIMES_SEMANTIC[i][:l], TIMES_CONTINUOUS[i][:l]))
    if results[i][-1] > max_p_value_times:
        max_p_value_times = results[i][-1]
    delta_times.append(round(np.mean(TIMES_CONTINUOUS[i][:l]) - np.mean(TIMES_SEMANTIC[i][:l]), 2))
    # print('Goal {} statistically significant, p={}, t_decstr={}, t_baseline={}'.format(i, results[i][-1],
    #                                                                                    round(np.mean(TIMES_SEMANTIC[i][:l]), 2),
    #                                                                                    round(np.mean(TIMES_CONTINUOUS[i][:l]), 2)
    #                                                                                    ))

results_distance = []
delta_distances = []
max_p_value_distance = -1
for i in range(N_GOALS):
    l = min(len(DISTANCES_SEMANTIC[i]), len(DISTANCES_CONTINUOUS[i]))
    results_distance.append(ttest_rel(DISTANCES_SEMANTIC[i][:l], DISTANCES_CONTINUOUS[i][:l]))
    if results_distance[i][-1] > max_p_value_distance:
        max_p_value_distance = results_distance[i][-1]
    delta_distances.append(round(np.mean(DISTANCES_CONTINUOUS[i][:l]) - np.mean(DISTANCES_SEMANTIC[i][:l]), 2))
    print('Goal {} statistically significant, p={}, t_decstr={}, t_baseline={}'.format(i, results_distance[i][-1],
                                                                                       round(np.mean(DISTANCES_SEMANTIC[i][:l]), 2),
                                                                                       round(np.mean(DISTANCES_CONTINUOUS[i][:l]), 2)
                                                                                       ))

# all_results = np.stack([delta_times, delta_distances])
# df = pd.DataFrame(all_results)
# df.to_csv('../opportunistic_study.csv', index=False)
# print('a')

def plot_time():
    TIMES_SEMANTIC = [[] for _ in range(5)]
    TIMES_CONTINUOUS = [[] for _ in range(5)]

    for i in range(DATA_LEN):
        # Only take into account successful episodes
        if str(data_decstr[i]['ag'][-1]) == str(data_decstr[i]['g'][0]):
            time_success_decstr = np.where(data_decstr[i]['rewards'] == 1.)[0][0]
            TIMES_SEMANTIC[data_decstr[i]['bucket']].append(time_success_decstr)
        # Same for baseline
        if data_baseline[i]['rewards'][-1] == 3.:
            time_success_decstr = np.where(data_baseline[i]['rewards'] == 3.)[0][0]
            TIMES_CONTINUOUS[data_decstr[i]['bucket']].append(time_success_decstr)

    ticks = ['Bucket {}'.format(i+1) for i in range(5)]

    plt.figure()

    bpl = plt.boxplot(TIMES_SEMANTIC, positions=np.array(range(len(TIMES_SEMANTIC)))*2.0-0.4, sym='')
    bpr = plt.boxplot(TIMES_CONTINUOUS, positions=np.array(range(len(TIMES_CONTINUOUS)))*2.0+0.4, sym='')
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')

    plt.plot([], c='#D7191C', label='DECSTR')
    plt.plot([], c='#2C7BB6', label='Continuous Baseline')
    plt.legend()

    plt.title('Box plot of time needed to reach a configuration')
    plt.ylabel('Timesteps')
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(0, 60)
    plt.grid()
    plt.show()


def get_moved_blocks():
    MOVED_BLOCKS_SEMANTIC = [[] for _ in range(5)]
    MOVED_BLOCKS_CONTINUOUS = [[] for _ in range(5)]

    for i in range(DATA_LEN):
        # Only take into account successful episodes
        nb_decstr = 0
        nb_continuous = 0
        if str(data_decstr[i]['ag'][-1]) == str(data_decstr[i]['g'][0]):
            for j in range(3):
                if np.linalg.norm(data_decstr[i]['obs'][-1][10 + 15 * j:13 + 15 * j] - data_decstr[i]['obs'][0][10 + 15 * j:13 + 15 * j]) > 0.01:
                    nb_decstr += 1
            MOVED_BLOCKS_SEMANTIC[data_decstr[i]['bucket']].append(nb_decstr)
        # Same for baseline
        if data_baseline[i]['rewards'][-1] == 3.:
            for j in range(3):
                if np.linalg.norm(data_baseline[i]['obs'][-1][10 + 15 * j:13 + 15 * j] - data_baseline[i]['obs'][0][10 + 15 * j:13 + 15 * j]) > 0.01:
                    nb_continuous += 1
            MOVED_BLOCKS_CONTINUOUS[data_decstr[i]['bucket']].append(nb_continuous)

    results_decstr = np.array([np.mean(e) for e in MOVED_BLOCKS_SEMANTIC])
    results_continuous = np.array([np.mean(e) for e in MOVED_BLOCKS_CONTINUOUS])

    std_decstr = np.array([np.std(e) for e in MOVED_BLOCKS_SEMANTIC])
    std_continuous = np.array([np.std(e) for e in MOVED_BLOCKS_CONTINUOUS])

    for i in range(5):
        print('===== Bucket {} ====='.format(i))
        print('DECSTR : {} += {} '.format(results_decstr[i], std_decstr[i]))
        print('Continuous : {} += {} '.format(results_continuous[i], std_continuous[i]))
