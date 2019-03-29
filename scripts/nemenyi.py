#!/usr/bin/env python
# coding: utf-8

# # Benchmark against state-of-the-art performance

# In[1]:


def friedman_test(*args):
    """
    source: http://tec.citius.usc.es/stac/doc/_modules/stac/nonparametric_tests.html#friedman_test

        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        -------
        F-value : float
            The computed F-value of the test.
        p-value : float
            The associated p-value from the F-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.

        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """

    import numpy as np   ###### MOVE import ?
    import scipy as sp
    import scipy.stats as st
    import itertools as it

    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row, reverse=True)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v)-1)/2. for v in row])

    rankings_avg = [sp.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r/sp.sqrt(k*(k+1)/(6.*n)) for r in rankings_avg]

    chi2 = ((12*n)/float((k*(k+1))))*((sp.sum(r**2 for r in rankings_avg))-((k*(k+1)**2)/float(4)))
    iman_davenport = ((n-1)*chi2)/float((n*(k-1)-chi2))

    p_value = 1 - st.f.cdf(iman_davenport, k-1, (k-1)*(n-1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def wilcoxon_test(score_A, score_B):

    import numpy as np
    import pandas as pd
    import math

    # compute abs delta and sign
    delta_score = [score_B[i] - score_A[i] for i in range(len(score_A))]
    sign_delta_score = list(np.sign(delta_score))
    abs_delta_score = list(map(abs, delta_score))

    # removing 0 values
    try:
        delta_score.remove(0.000000)
        sign_delta_score.remove(0.000000)
        abs_delta_score.remove(0.000000)
    except:
        pass

    N_r = float(len(delta_score))

    # hadling scores
    score_df = pd.DataFrame({'abs_delta_score':abs_delta_score, 'sign_delta_score':sign_delta_score })

    # sort
    score_df.sort_values(by='abs_delta_score', inplace=True)
    score_df.index = range(1,len(score_df)+1)

    # adding ranks
    score_df['Ranks'] = score_df.index
    score_df['Ranks'] = score_df['Ranks'].astype('float64')

    score_df.dropna(inplace=True)

    # z : pouput value
    W = sum(score_df['sign_delta_score'] * score_df['Ranks'])
    z = W/(math.sqrt(N_r*(N_r+1)*(2*N_r+1)/6.0))

    # rejecte or not the null hypothesis
    null_hypothesis_rejected = False
    if z < -1.96 or z > 1.96:
        null_hypothesis_rejected = True

    return z, null_hypothesis_rejected


def nemenyi_test(param_bench, only_best_snb=False, min_trainset_size = 0, plot=True):
    """
    implementation of the nemenyi test
    """

    import pandas as pd
    import numpy as np
    import Orange
    import math
    import matplotlib.pyplot as plt

    # setting of this function


    # I - Load results
    # ----------------

    # Load competitors data
    with open(param_bench["input_dir"]+"/all_sota_models_accuracy.csv", 'r') as fp:
        sota_df = pd.read_csv(fp, index_col=0)

    # Load result of benchMarck
    with open(param_bench["results_dir"]+"/scores_all_expe_all_dataset.csv", 'r') as fp:
        snbs_df = pd.read_csv(fp, index_col=0)

    # Load NN models data
    with open(param_bench["input_dir"]+"/all_NN_models_accuracy.csv", 'r') as fp: ### TODO : construire le path a partir du dico en param
        nn_df = pd.read_csv(fp, index_col=0)


    # II - cleaning results data tables
    # ---------------------------------

    keept_NN_approaches = ["ResNet","FCN"]

    # include all snb or not in or benchmark
    if only_best_snb:
        snbs_df.drop(list(set(snbs_df.columns) - {"Best_CG_Score"}), axis=1, inplace=True)  # "Best_CG_Score"
        snbs_df.columns = ["FEARS"]

    # removing the columns that gives the name of the Best exeperiment according to its compression gain
    if "Best_CG_Expe" in snbs_df.columns:
        snbs_df.drop(['Best_CG_Expe'],axis=1, inplace=True)

    # keeping only resnet that is the best NN approach
    nn_df.drop(list(set(nn_df.columns) - set(keept_NN_approaches)),axis=1, inplace=True)


    # III - preparing final results dataset
    # -------------------------------------

    # concat dataframe
    fullbench_df = pd.concat([sota_df, nn_df, snbs_df], axis=1 , sort=True)

    # filtering data by size
    fullbench_df = fullbench_df[fullbench_df['N Train'] > min_trainset_size]

    nb_dataset = fullbench_df.shape

    # cleanning : removing of the columns that dont represent a clasifier
    fullbench_df.drop(["Classes","N Train","N test","length n","Type","SNB"],axis=1, inplace=True)

    fullbench_df.rename(columns={'COTE (ensemble)':'COTE', 'EE (PROP)':'EE','Learning Shapelet (LS)':'LS'}, inplace=True)

    # removing datasets (row) forwhich there is no MODL experiemnts
    fullbench_df.dropna(inplace=True)

    # IV - Nemenyi test
    # -----------------

    # Control of the size of the results table
    if True : #fullbench_df.shape[0] > 15 and fullbench_df.shape[1] > 4 and fullbench_df.shape[1] < 21 : # we need at least 11 remainning dataset and 6 algorithms to compare (Orange.evaluation bug avec plus de 20 classifieurs)

        classifiers = list(fullbench_df.columns)

        score_arrays = []
        for classifier in classifiers:
            score_arrays.append(fullbench_df[classifier].tolist())

        iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(*score_arrays) # TODO : condition pour numm hypothesis

        # if the null hypothesis is rejected (i.e. there is not significant different beteew algorithms)
        if p_value < 0.05:

            # Returns critical difference for Nemenyi
            cd = Orange.evaluation.scoring.compute_CD(avranks=rankings_avg, n=len(fullbench_df), alpha="0.05", test="nemenyi")

            # Save the nemenyi plot

            if only_best_snb:
                suffix = "BestSNB"
            else:
                suffix = "AllSNB"

            rank_viz_fpath = param_bench["results_dir"]+"/nemenyi_N_min_"+str(min_trainset_size)+"_"+suffix+".png"
            lowv = math.floor(min(rankings_avg))
            highv = math.ceil(max(rankings_avg))
            width = (highv - lowv)*1.2+2

            if plot:
                Orange.evaluation.scoring.graph_ranks(
                    filename=rank_viz_fpath,
                    avranks=rankings_avg,
                    names=classifiers,
                    cd=cd,
                    lowv=lowv,
                    highv=highv,
                    width=width,
                    fontsize=15,
                    textspace=4)
                plt.close()

            # les test de wilcoxon paire a paire ....

            if only_best_snb:

                # ------- TODO : parcourir les classifieurs dans l'ordre !!

                # get the ordered list of clssifiers
                prov_df = pd.DataFrame({'classifiers':classifiers, 'rankings_avg':rankings_avg })
                prov_df.sort_values(by='rankings_avg', inplace=True)
                ordered_classifiers = prov_df['classifiers'].tolist()

                # build the matrix of pairewide comparison
                pairewise_comparison_df = pd.DataFrame(
                        np.zeros(shape=(len(ordered_classifiers),len(ordered_classifiers))),
                        columns=ordered_classifiers,
                        index=ordered_classifiers
                        )

                # pairewide comparison
                for classifier_col in ordered_classifiers:
                    for classifier_row in ordered_classifiers:
                        if classifier_col != classifier_row:
                            z, null_hypothesis_rejected = wilcoxon_test(fullbench_df[classifier_col].tolist(), fullbench_df[classifier_row].tolist())
                            if null_hypothesis_rejected:
                                pairewise_comparison_df[classifier_col][classifier_row] = 1

                # save pairewide comparison
                #pairewise_comparison_df.to_csv(path_or_buf=param_bench["results_dir"]+"/wilcoxon_pairewise_N_min_"+str(min_trainset_size)+"_"+suffix+".csv")

                # build graphic showing all the pairewise comparisons
                if plot:
                    #plt.clf()
                    plt.imshow(pairewise_comparison_df, cmap=plt.cm.gray)
                    plt.xticks(np.arange(0,len(pairewise_comparison_df)), pairewise_comparison_df.columns, rotation='vertical', fontsize=15)
                    plt.yticks(np.arange(0,len(pairewise_comparison_df)), pairewise_comparison_df.columns, fontsize=15)
                    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)

                    plt.savefig(param_bench["results_dir"]+"/wilcoxon_pairewise_N_min_"+str(min_trainset_size)+"_"+suffix+".png", bbox_inches='tight')
                    plt.close()

        else:
            print("No Nemenyi plot have been generated for N_train_min = "+str(min_trainset_size)+" : there is no significant difference between the learning algorithms !")
    else:
        print("No Nemenyi plot have been generated for N_train_min = "+str(min_trainset_size)+" : not enought dataset remaining !")

    return classifiers, rankings_avg, nb_dataset


def wrapper_nemenyi_test(param_bench):

    for min_N in param_bench["nemenyi_min_trainset_size"]:
        #nemenyi_test(param_bench, only_best_snb=False, min_trainset_size = min_N)
        nemenyi_test(param_bench, only_best_snb=True, min_trainset_size = min_N)


def wrapper_ranks_vs_Ntrain(param_bench):

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.gridspec as gridspec

    filtered_clf = ["MODL"]

    N = range(0,1500,100)
    Rank = []

    for min_N in N:
        classifiers, rankings_avg, shape = nemenyi_test(param_bench, only_best_snb=True, min_trainset_size = min_N, plot=False)



        for clf in filtered_clf:
            id = classifiers.index(clf)
            Rank.append(rankings_avg[id])
            print("Min size = "+str(min_N)+"   rank = "+str(rankings_avg[id])+"   shape = "+str(shape))

    plt.plot(N,Rank, color='black')
    plt.ylim(8,3.5)  # decreasing time
    plt.xlabel('Min N', fontsize=15)
    plt.ylabel('Avg rank', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(param_bench["results_dir"]+"/rank_MODL_vs_Min_N_Train.png", bbox_inches='tight')
    plt.close()

def wrapper_delaited_results(param_bench):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # ['name_best_model', 'test_AUC', 'test_ACC', 'nb_informative_var','nb_selected_representations', 'selected_representations',
    #'selected_shema', 'nb_loop_rep_selection', 'run_time_on_dataset']
    with open(param_bench["results_dir"]+"/detailed_results_best_models.csv", 'r') as fp:
        detailed_results_df = pd.read_csv(fp, index_col=0)

    detailed_results_df['run_time_on_dataset'] = detailed_results_df['run_time_on_dataset']/3600. # time in hours

    interesting_col_name = ['nb_informative_var','nb_selected_representations','nb_loop_rep_selection','run_time_on_dataset','selected_shema']

    for col in interesting_col_name:

        nb_bins = min(10,len(detailed_results_df[col].unique()))

        # alignement des tiks selon le graph
        align_tiks='mid'
        if col == "nb_selected_representations":
            align_tiks='mid' #'left'

        input_bins = np.arange(1,nb_bins+2) - 0.5
        n, bins, patches = plt.hist(x=detailed_results_df[col].tolist(), bins=input_bins, color='silver', align=align_tiks)
        #plt.xlabel('Value')
        #plt.ylabel('Frequency')
        plt.locator_params(axis='x', nbins=nb_bins)
        plt.xticks(range(1, nb_bins+1), fontsize=15)
        plt.yticks(fontsize=15)
        #plt.title(col)
        maxfreq = n.max()
        #plt.xlim([-1, nb_bins])
        plt.savefig(param_bench["results_dir"]+"/histogram_"+col+".png", bbox_inches='tight')
        plt.close()


def wrapper_use_of_rep_plot(param_bench):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # ['name_best_model', 'test_AUC', 'test_ACC', 'nb_informative_var','nb_selected_representations', 'selected_representations',
    #'selected_shema', 'nb_loop_rep_selection', 'run_time_on_dataset']
    with open(param_bench["results_dir"]+"/detailed_results_best_models.csv", 'r') as fp:
        detailed_results_df = pd.read_csv(fp, index_col=0)

    for rep in param_bench["representations"]:
        detailed_results_df[rep] = detailed_results_df["selected_representations"].apply(lambda x: 1 if rep in x else 0)

    count_by_rep = []
    for rep in param_bench["representations"]:
        count_by_rep.append(detailed_results_df[rep].sum())



    # Pie chart
    labels = param_bench["representations"]
    sizes = count_by_rep
    # only "explode" the 2nd slice (i.e. 'Hogs')
    #explode = (0, 0.1, 0, 0)
    #add colors
    colors = ['#EFEFEF','#D1D1D1','#B9B8B8','#999999','#7A7A7A','#5A5A5A','#000000']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors,shadow=False, startangle=90, textprops={'fontsize': 15})

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.savefig(param_bench["results_dir"]+"/pie_rep.png", bbox_inches='tight')
    plt.close()

def get_single_rep_acc_by_dataset(param_bench,onTestSet=True):
    """
    Computes accuracy for a given dataset, by a given model parameters (representations, schema)
    Use check_experiments_runs() to verify which experiments parameters are available in the interim data dir
    """

    from sklearn.metrics import accuracy_score
    from load_data import load_data_from_dir
    from load_data import load_y_pred

    accuracy_scores = {}
    for dataset_name in param_bench['dataset_names']:
        if onTestSet:
            _, test_df = load_data_from_dir(dataset_name, rootdir=param_bench['input_dir'])
        else:
            test_df,_ = load_data_from_dir(dataset_name, rootdir=param_bench['input_dir'])

        y_test = test_df[0].tolist()

        for label_pred_dict in load_y_pred(dataset_name, rootdir=param_bench['output_dir']+"/single_rep_expe"):
            y_pred = label_pred_dict['preds']
            accuracy_scores[dataset_name] = accuracy_score(y_test, y_pred)
            break

    return accuracy_scores



def wrapper_acc_single_rep_plot(param_bench):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    acc_single_rep = get_single_rep_acc_by_dataset(param_bench)
    acc_single_rep_df = pd.DataFrame.from_dict(acc_single_rep, orient='index', columns=['ACC_single_rep'])

    with open(param_bench["results_dir"]+"/detailed_results_best_models.csv", 'r') as fp:
        df = pd.read_csv(fp, index_col=0)

    df.drop(list(set(df.columns) - {"test_ACC"}), axis=1, inplace=True)

    df = pd.concat([df, acc_single_rep_df], axis=1 , sort=True)

    df = df.dropna() #### TODO : quelques jeu de donnes ou l'expe n'a pas ete faite !!

    #plt.scatter(x,y,s=100)
    plt.scatter(df['ACC_single_rep'].tolist(),df['test_ACC'].tolist(),s=6 , color='black')
    plt.plot(np.arange(0, 1.1, 0.1),np.arange(0, 1.1, 0.1), '--', linewidth=1, color='gray')
    #plt.title('Nuage de points avec Matplotlib')
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.xlabel('ACC best single representation', fontsize=15)
    plt.ylabel('ACC', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axes = plt.gca()
    axes.set_xlim([0.2,1])
    axes.set_ylim([0.2,1])
    plt.savefig(param_bench["results_dir"]+"/ACC_single_rep_vs_ACC.png", bbox_inches='tight')
    plt.close()

    df['count_best'] = df.apply(lambda row: 1 if row['test_ACC'] > row['ACC_single_rep'] else 0, axis=1)

    best_rate = df['count_best'].sum()/len(df)

    print("best rate = "+str(best_rate))

    z, null_hypothesis_rejected = wilcoxon_test(df['test_ACC'].tolist(), df['ACC_single_rep'].tolist())

    print('Wilcoxon z = '+str(z))

########

def wrapper_acc_single_rep_plot_trainSet(param_bench):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    acc_single_rep = get_single_rep_acc_by_dataset(param_bench, onTestSet=False)
    acc_single_rep_df = pd.DataFrame.from_dict(acc_single_rep, orient='index', columns=['ACC_single_rep'])

######## :-/

    with open(param_bench["results_dir"]+"/detailed_results_best_models.csv", 'r') as fp:
        df = pd.read_csv(fp, index_col=0)

    df.drop(list(set(df.columns) - {"test_ACC"}), axis=1, inplace=True)

    df = pd.concat([df, acc_single_rep_df], axis=1 , sort=True)

    df = df.dropna() #### TODO : quelques jeu de donnes ou l'expe n'a pas ete faite !!

    #plt.scatter(x,y,s=100)
    plt.scatter(df['ACC_single_rep'].tolist(),df['test_ACC'].tolist(),s=6 , color='black')
    plt.plot(np.arange(0, 1.1, 0.1),np.arange(0, 1.1, 0.1), '--', linewidth=1, color='gray')
    #plt.title('Nuage de points avec Matplotlib')
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.xlabel('ACC best single representation', fontsize=15)
    plt.ylabel('ACC', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axes = plt.gca()
    axes.set_xlim([0.2,1])
    axes.set_ylim([0.2,1])
    plt.savefig(param_bench["results_dir"]+"/ACC_single_rep_vs_ACC.png", bbox_inches='tight')
    plt.close()

    df['count_best'] = df.apply(lambda row: 1 if row['test_ACC'] > row['ACC_single_rep'] else 0, axis=1)

    best_rate = df['count_best'].sum()/len(df)

    print("best rate = "+str(best_rate))

    z, null_hypothesis_rejected = wilcoxon_test(df['test_ACC'].tolist(), df['ACC_single_rep'].tolist())

    print('Wilcoxon z = '+str(z))



def wrapper_running_time_detailed_plot(param_bench):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    with open(param_bench["results_dir"]+"/detailed_results_best_models.csv", 'r') as fp:
        df = pd.read_csv(fp, index_col=0)

    with open(param_bench["input_dir"]+"/all_sota_models_accuracy.csv", 'r') as fp:
        df_2 = pd.read_csv(fp, index_col=0)

    df = pd.concat([df, df_2], axis=1 , sort=True)
    df["nb_points"] = df['N Train']*df['length n']

    plt.scatter(df['N Train'].tolist(),df['run_time_on_dataset'].tolist(),s=6 , color='black')
    #plt.plot(np.arange(10, 1, 200),np.arange(10, 1, 200), '--', linewidth=1, color='gray')
    #plt.title('Nuage de points avec Matplotlib')
    plt.xlabel('N', fontsize=15)
    plt.ylabel('Runing time (s)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axes = plt.gca()
    axes.set_xscale('log')
    axes.set_yscale('log')
    #plt.grid(True,which="both",ls="-")
    #axes.set_xlim([0.2,1])
    #axes.set_ylim([0.2,1])
    plt.savefig(param_bench["results_dir"]+"/runing_tim_vs_N.png", bbox_inches='tight')
    plt.close()







# Compute Nemenyi
# cf https://github.com/biolab/orange3/blob/master/Orange/evaluation/scoring.py


# exemple unitaire

# import Orange
# import matplotlib.pyplot as plt
# names = ["first", "third", "second", "fourth" ]
# avranks =  [1.9, 3.2, 2.8, 3.3 ]
# cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
# plt.show()
