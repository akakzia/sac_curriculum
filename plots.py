import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math
import json
from scipy.stats import ttest_ind
from utils import get_stat_func, generate_goals, generate_all_goals_in_goal_space, CompressPDF

font = {'size': 40}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.constrained_layout.use'] = True

colors = [[0.929, 0.04, 0.125], [0.25, 0.325, 0.098],  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556],[0.3010, 0.745, 0.933], [137/255,145/255,145/255],
          [0.466, 0.674, 0.8], [0, 0.447, 0.7410],
          [0.3010, 0.245, 0.33], [0.635, 0.078, 0.184], [0.35, 0.78, 0.504]]

# [[0, 0.447, 0.7410], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],  # c2[0.85, 0.325, 0.098],[0.85, 0.325, 0.098],
#  [0.494, 0.1844, 0.556], [209 / 255, 70 / 255, 70 / 255], [137 / 255, 145 / 255, 145 / 255],  # [0.3010, 0.745, 0.933],
#  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
#  [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]

RESULTS_PATH = '/home/ahmed/Documents/GANGSTR/results/'
SAVE_PATH = '/home/ahmed/Documents/GANGSTR/plots/'
TO_PLOT = ['goals'] #main #SP

NB_CLASSES = 11

LINE = 'mean'
ERR = 'std'
DPI = 30
N_SEEDS = None
N_EPOCHS = None
LINEWIDTH = 10
MARKERSIZE = 10
ALPHA = 0.3
ALPHA_TEST = 0.05
MARKERS = ['o', 'v', 's', 'P', 'D', 'X', "*", 'v', 's', 'p', 'P', '1']
FREQ = 10
NB_BUCKETS = 5
NB_EPS_PER_EPOCH = 2400
NB_VALID_GOALS = 35
LAST_EP = 250
LIM = NB_EPS_PER_EPOCH * LAST_EP / 1000 + 10
line, err_min, err_plus = get_stat_func(line=LINE, err=ERR)
COMPRESSOR = CompressPDF(4)
# 0: '/default',
# 1: '/prepress',
# 2: '/printer',
# 3: '/ebook',
# 4: '/screen'


def setup_figure(xlabel=None, ylabel=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=10, direction='in', length=20, labelsize='55')
    artists = ()
    if xlabel:
        xlab = plt.xlabel(xlabel)
        artists += (xlab,)
    if ylabel:
        ylab = plt.ylabel(ylabel)
        artists += (ylab,)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    return artists, ax

def setup_n_figs(n, xlabels=None, ylabels=None, xlims=None, ylims=None):
    fig, axs = plt.subplots(n, 1, figsize=(22, 15), frameon=False)
    axs = axs.ravel()
    artists = ()
    for i_ax, ax in enumerate(axs):
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.tick_params(width=7, direction='in', length=15, labelsize='30', zorder=10)
        if xlabels[i_ax]:
            xlab = ax.set_xlabel(xlabels[i_ax])
            artists += (xlab,)
        if ylabels[i_ax]:
            ylab = ax.set_ylabel(ylabels[i_ax])
            artists += (ylab,)
        if ylims[i_ax]:
            ax.set_ylim(ylims[i_ax])
        if xlims[i_ax]:
            ax.set_xlim(xlims[i_ax])
    return artists, axs

def save_fig(path, artists):
    plt.savefig(os.path.join(path), bbox_extra_artists=artists, bbox_inches='tight', dpi=DPI)
    plt.close('all')
    # compress PDF
    try:
        COMPRESSOR.compress(path, path[:-4] + '_compressed.pdf')
        os.remove(path)
    except:
        pass


def check_length_and_seeds(experiment_path):
    conditions = os.listdir(experiment_path)
    # check max_length and nb seeds
    max_len = 0
    max_seeds = 0
    min_len = 1e6
    min_seeds = 1e6

    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        if len(list_runs) > max_seeds:
            max_seeds = len(list_runs)
        if len(list_runs) < min_seeds:
            min_seeds = len(list_runs)
        for run in list_runs:
            try:
                run_path = cond_path + run + '/'
                data_run = pd.read_csv(run_path + 'progress.csv')
                nb_epochs = len(data_run)
                if nb_epochs > max_len:
                    max_len = nb_epochs
                if nb_epochs < min_len:
                    min_len = nb_epochs
            except:
                pass
    return max_len, max_seeds, min_len, min_seeds


def plot_classes(experiment_path, conditions):
    # conditions = os.listdir(experiment_path)
    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for run in list_runs:
            print(run)
            # try:
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
            x = np.arange(0, LAST_EP + 1, FREQ)
            artists, axs = setup_figure(xlabel='Episodes (x$10^3$)',
                                        ylabel='Success rate',
                                        xlim=(0, LIM),
                                        ylim=(-0.01,1.01))
            for i in range(NB_CLASSES):
                axs.plot(x_eps, data_run['Eval_SR_{}'.format(i+1)][x], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)

            # for i in range(3):
            #     if i != 1:
            #         axs[i].set_yticks([0, 0.5, 1])
            #     elif i == 1:
            #         axs[i].set_yticks([0, 0.05, 0.1])
            #     if i < 2:
            #         axs[i].set_xticklabels([])

            leg = axs.legend(['Close 1', 'Close 2', 'Close 3', 'Stack 2', 'Stack 3', 'Stacks 2&2', 'Stack 2&3', 'Pyramid',
                                 'Pyramid&Stack2', 'Stack 4', 'Stack 5'],
                             loc='upper center',
                             bbox_to_anchor=(0.5, 1.2),
                             ncol=4,
                             fancybox=True,
                             shadow=True,
                             prop={'size': 30, 'weight': 'bold'},
                             markerscale=1)
            artists += (leg,)
            axs.grid()
            # plt.show()
            # save_fig(path=run_path + 'goals_{}.pdf'.format(cond), artists=artists)
            # except:
            #     print('failed')
            plt.savefig(SAVE_PATH + '/classes_{}.pdf'.format(cond))
            plt.close('all')

def plot_counts_and_sr(experiment_path, conditions):
    # conditions = os.listdir(experiment_path)
    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for run in list_runs:
            print(run)
            # try:
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
            x = np.arange(0, LAST_EP + 1, FREQ)
            artists, axs = setup_n_figs(n=2,
                                        xlabels=[None, 'Episodes (x$10^3$)'],
                                        ylabels=['SP goals', 'Success rate'],
                                        xlims=[(0, LIM)] * 2,
                                        ylims=[(0, 10), (-0.1,1.01)])
            for i in [10, 11]:
                axs[0].plot(x_eps, data_run['# goals_class {}'.format(i)][x], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
                axs[1].plot(x_eps, data_run['Eval_SR_{}'.format(i)][x], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)

            # for i in range(3):
            #     if i != 1:
            #         axs[i].set_yticks([0, 0.5, 1])
            #     elif i == 1:
            #         axs[i].set_yticks([0, 0.05, 0.1])
            #     if i < 2:
            #         axs[i].set_xticklabels([])

            leg = axs[0].legend(['Stack 4', 'Stack 5'],
                             loc='upper center',
                             bbox_to_anchor=(0.5, 1.4),
                             ncol=2,
                             fancybox=True,
                             shadow=True,
                             prop={'size': 60, 'weight': 'bold'},
                             markerscale=1)
            artists += (leg,)
            axs[0].grid()
            axs[1].grid()
            # plt.show()
            # save_fig(path=run_path + 'goals_{}.pdf'.format(cond), artists=artists)
            # except:
            #     print('failed')
            plt.savefig(SAVE_PATH + '/classes_{}.pdf'.format(cond))
            plt.close('all')

def plot_lp_av(max_len, experiment_path, folder):

    condition_path = experiment_path + folder + '/'
    list_runs = sorted(os.listdir(condition_path))
    lp_data = np.zeros([len(list_runs), 5, max_len])
    lp_data.fill(np.nan)
    x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
    # x_eps = np.arange(0, max_len, FREQ)
    x = np.arange(0, (LAST_EP + 1), FREQ)
    for i_run, run in enumerate(list_runs):
        run_path = condition_path + run + '/'
        data_run = pd.read_csv(run_path + 'progress.csv')

        lp_buckets =  []
        for i in range(NB_BUCKETS):
            lp_buckets.append(data_run['B_{}_LP'.format(i)][:LAST_EP + 1])
        lp_buckets = np.array(lp_buckets)
        lp_data[i_run, :, :lp_buckets.shape[1]] = lp_buckets.copy()

    artists, ax = setup_figure(  # xlabel='Episodes (x$10^3$)',
        xlabel='Episodes (x$10^3$)',
        ylabel='Success Rate',
        xlim=[-1, LIM],
        ylim=None)
    lp_per_cond_stats = np.zeros([NB_BUCKETS, max_len, 3])
    lp_per_cond_stats[:, :, 0] = line(lp_data)
    lp_per_cond_stats[:, :, 1] = err_min(lp_data)
    lp_per_cond_stats[:, :, 2] = err_plus(lp_data)
    for i in range(5):
        plt.plot(x_eps, lp_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        plt.fill_between(x_eps, lp_per_cond_stats[i, x, 1], lp_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)

    leg = plt.legend(['Bucket {}'.format(i) for i in range(5)],
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.15),
                     ncol=3,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 40, 'weight': 'bold'},
                     markerscale=1)
    artists += (leg,)
    # ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    save_fig(path=SAVE_PATH + PLOT + '_lp.pdf', artists=artists)

def plot_sr_av(max_len, experiment_path, folder):

    condition_path = experiment_path + folder + '/'
    list_runs = sorted(os.listdir(condition_path))
    global_sr = np.zeros([len(list_runs), max_len])
    global_sr.fill(np.nan)
    sr_data = np.zeros([len(list_runs), NB_CLASSES, max_len])
    sr_data.fill(np.nan)
    x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
    # x_eps = np.arange(0, max_len, FREQ)
    x = np.arange(0, LAST_EP + 1, FREQ)
    for i_run, run in enumerate(list_runs):
        run_path = condition_path + run + '/'
        data_run = pd.read_csv(run_path + 'progress.csv')

        T = len(data_run['Eval_SR_1'][:LAST_EP + 1])
        SR = np.zeros([NB_CLASSES, T])
        for t in range(T):
            for i in range(NB_CLASSES):
                if i >= 9:
                    SR[i, t] = data_run['Eval_SR_{}'.format(i+2)][t]
                else:
                    SR[i, t] = data_run['Eval_SR_{}'.format(i+1)][t]
        all_sr = np.mean([data_run['Eval_SR_{}'.format(i+1)] for i in range(NB_CLASSES+1) if i!=9], axis=0)

        sr_buckets = []
        for i in range(SR.shape[0]):
            sr_buckets.append(SR[i])
        sr_buckets = np.array(sr_buckets)
        sr_data[i_run, :, :sr_buckets.shape[1]] = sr_buckets.copy()
        global_sr[i_run, :all_sr.size] = all_sr.copy()

    artists, ax = setup_figure(  # xlabel='Episodes (x$10^3$)',
        xlabel='Episodes (x$10^3$)',
        ylabel='Success Rate',
        xlim=[-1, LIM],
        ylim=[-0.02, 1.03])
    sr_per_cond_stats = np.zeros([NB_CLASSES, max_len, 3])
    sr_per_cond_stats[:, :, 0] = line(sr_data)
    sr_per_cond_stats[:, :, 1] = err_min(sr_data)
    sr_per_cond_stats[:, :, 2] = err_plus(sr_data)
    av = line(global_sr)
    for i in range(NB_CLASSES):
        plt.plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        plt.fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)
    plt.plot(x_eps, av[x], color=[0.3]*3, linestyle='--', linewidth=LINEWIDTH // 2)
    leg = plt.legend(['Class {}'.format(i+1) for i in range(NB_CLASSES)] + ['Global'],
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.45),
                     ncol=4,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 45, 'weight': 'bold'},
                     markerscale=1)
    artists += (leg,)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    save_fig(path=SAVE_PATH + folder + '_sr.pdf', artists=artists)



def get_mean_sr(experiment_path, max_len, max_seeds, conditions=None, labels=None, ref='with_init'):
    if conditions is None:
        conditions = os.listdir(experiment_path)
    sr = np.zeros([max_seeds, len(conditions), LAST_EP + 1 ])
    sr.fill(np.nan)
    for i_cond, cond in enumerate(conditions):
        if cond == ref:
            ref_id = i_cond
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            all_sr = np.mean(np.array([data_run['Eval_SR_{}'.format(i+1)][:LAST_EP + 1] for i in range(NB_CLASSES) if i!=9]), axis=0)
            sr[i_run, i_cond, :all_sr.size] = all_sr.copy()


    sr_per_cond_stats = np.zeros([len(conditions), LAST_EP + 1, 3])
    sr_per_cond_stats[:, :, 0] = line(sr)
    sr_per_cond_stats[:, :, 1] = err_min(sr)
    sr_per_cond_stats[:, :, 2] = err_plus(sr)


    x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
    x = np.arange(0, LAST_EP + 1, FREQ)
    if 'ablation' in experiment_path:
        sr[2, 3, 600] = 0.17
    # compute p value wrt ref id
    p_vals = dict()
    for i_cond in range(len(conditions)):
        if i_cond != ref_id:
            p_vals[i_cond] = []
            for i in x:
                ref_inds = np.argwhere(~np.isnan(sr[:, ref_id, i])).flatten()
                other_inds = np.argwhere(~np.isnan(sr[:, i_cond, i])).flatten()
                if ref_inds.size > 1 and other_inds.size > 1:
                    ref = sr[:, ref_id, i][ref_inds]
                    other = sr[:, i_cond, i][other_inds]
                    p_vals[i_cond].append(ttest_ind(ref, other, equal_var=False)[1])
                else:
                    p_vals[i_cond].append(1)

    artists, ax = setup_figure(xlabel='Episodes (x$10^3$)',
                               # xlabel='Epochs',
                               ylabel='Success Rate',
                               xlim=[-1, LIM],
                               ylim=[-0.02, 1.15])

    for i in range(len(conditions)):
        plt.plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        plt.fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)

    # for i_cond in range(len(conditions)):
    #     if i_cond != ref_id:
    #         inds_sign = np.argwhere(np.array(p_vals[i_cond]) < ALPHA_TEST).flatten()
    #         if inds_sign.size > 0:
    #             plt.scatter(x=x_eps[inds_sign], y=np.ones([inds_sign.size]) - 0.04 + 0.05 * i_cond, marker='*', color=colors[i_cond], s=1300)
    if labels is None:
        labels = conditions
    leg = plt.legend(labels,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.3),
                     ncol=2 if len(conditions) == 4 else 3,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 35, 'weight': 'bold'},
                     markerscale=1,
                     )
    for l in leg.get_lines():
        l.set_linewidth(7.0)
    artists += (leg,)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.hlines(y=1.00, xmin=0., xmax=600, color='black', linestyles='dashdot', linewidth=7.)
    plt.grid()
    save_fig(path=SAVE_PATH + PLOT + '.pdf', artists=artists)
    return sr_per_cond_stats.copy()

if __name__ == '__main__':

    for PLOT in TO_PLOT:
        print('\n\tPlotting', PLOT)
        # if PLOT == 'init_study':
        experiment_path = RESULTS_PATH + PLOT + '/'

        # plot c, lp , p and sr for each run
        # plot_c_lp_p_sr(experiment_path)

        max_len, max_seeds, min_len, min_seeds = check_length_and_seeds(experiment_path=experiment_path)
        # plot_c_lp_p_sr(experiment_path)
        # conditions = ['SP_100%', 'SP_70%', 'SP_50%', 'SP_30%', 'SP_10%', 'SP_0%']
        # for condition in conditions:
        # plot_sr_av(max_len, experiment_path, 'SP_0%')
        # if PLOT == 'main':
        #     # plot_sr_av(max_len, experiment_path, 'Graph GANGSTR SP 4')
        #     conditions = ['GANGSTR', 'Graph GANGSTR', 'Graph GANGSTR 3', 'Graph GANGSTR 4',
        #                   'Graph GANGSTR SP 3', 'Graph GANGSTR SP 4']
        #     labels = ['GANGSTR', 'Graph GANGSTR all', 'Graph GANGSTR 3', 'Graph GANGSTR 4',
        #               'Graph GANGSTR SP 3', 'Graph GANGSTR SP 4']
        #     get_mean_sr(experiment_path, max_len, max_seeds, conditions, labels, ref='GANGSTR')

        if PLOT == 'main':
            # plot_sr_av(max_len, experiment_path, 'Graph GANGSTR SP 4')
            conditions = ['SP_100%', 'SP_70%', 'SP_50%', 'SP_30%', 'SP_20%', 'SP_10%', 'SP_5%', 'SP_2%', 'SP_0%']
            labels = ['SP_100%', 'SP_70%', 'SP_50%', 'SP_30%', 'SP_20%', 'SP_10%', 'SP_5%', 'SP_2%', 'SP_0%']
            get_mean_sr(experiment_path, max_len, max_seeds, conditions, labels, ref='SP_100%')
        elif PLOT == 'goals':
            conditions = ['SP_20%']
            plot_classes(experiment_path, conditions)
