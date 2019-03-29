
# = "C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\interim"
def check_experiments_runs(data_dpath):
    """
    prints params for exp that have ran on all datasets
    """
    import os
    import simplejson as json
    from load_data import get_dataset_names

    # List all configs available in exp results per dataset


    json_fpath_list = []
    all_dataset_names = get_dataset_names()
    for dname in all_dataset_names:
        dpath = f"{data_dpath}\\{dname}"

        for run_dname in os.listdir(dpath):
            for fname in os.listdir(f"{dpath}\\{run_dname}"):
                if fname.endswith('.json'):
                    json_fpath = f"{dpath}\\{run_dname}\\{fname}"
                    json_fpath_list.append(json_fpath)

    # Create dict of params
    exp_params_dict = {}
    for json_fpath in json_fpath_list:
        with open(json_fpath, 'r') as fp:
            json_content = json.load(fp)

            params = json_content['params']

            params_hashable = {
                'schema' : params['schema'],
                'n_features' : params['n_features']
                }

            params_hashable['representations'] = frozenset(params['representations'])
            param_key = frozenset(params_hashable.items())

            if param_key not in exp_params_dict.keys():
                exp_params_dict[param_key] = [json_content['dataset_name']]
            else:
                exp_params_dict[param_key].append(json_content['dataset_name'])

    # Check exp params where all datasets have been processed
    for k, v in exp_params_dict.items():
        if set(v) == set(all_dataset_names):
            print(k)

def avg_auc(y_test, y_score):
    """
    Compute average per-class AUC
    y_test is list of actual labels
    y_score is list of list where inner list is confidence score given by classifier for each label
    """

    # Get label info
    classes = sorted(list(set(y_test)))
    n_classes = len(classes)

    # Ensure y_score is np.array with adequat shape
    import numpy as np
    y_score = np.array(y_score)
    assert n_classes==y_score.shape[1],\
    "Number of unique label values is inconsistent with score input shape"

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score, roc_curve, auc

    # Case: binary label
    if n_classes == 2:
        y_score = np.array(y_score[:, 1], dtype='float64')
        y_test = np.array(y_test, dtype='int32')

        auc = roc_auc_score(y_test, y_score)

    # Case: multilabels
    else:
        # Binarize labels
        y_test = label_binarize(y_test, classes=classes)

        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Get avg AUC by class (implement avg)
        auc = np.mean(list(roc_auc.values()))

    return auc


# = "C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\interim"
def get_cg(param_exp):
    """
    Parse the Compression Gain score for a given model on a given train dataset
    eg:
        dataset_name :  'Adiac'
        representations : ['TS', 'D', 'DD', 'CUMSUM', 'DCUMSUM', 'ACF', 'PS']
        schema : 0
        n_features = 20000
        n_trees = 100

    Use check_experiments_runs() to verify which experiments parameters are available in the interim data dir
    """
    import os
    import simplejson as json
    import pandas as pd

    dataset_name = param_exp['dataset_name']
    representations = set(param_exp['representations'])
    schema = param_exp['schema']
    n_features = param_exp['n_features']
    n_trees = param_exp['n_trees']
    root_dir_interim = param_exp['output_dir']

    for run_dname in os.listdir(f"{root_dir_interim}/{dataset_name}"):
        run_dpath = f"{root_dir_interim}/{dataset_name}/{run_dname}"
        json_fname = [fname for fname in os.listdir(run_dpath) if fname.endswith('.json')][0]
        json_fpath = f"{run_dpath}/{json_fname}"
        with open(json_fpath, 'r') as fp:
            json_content = json.load(fp)

            ref_representations = set(json_content['params']['representations'])
            ref_schema = int(json_content['params']['schema'])
            ref_n_features = int(json_content['params']['n_features'])
            ref_n_trees = int(json_content['params']['n_trees'])

            #if (ref_representations == representations and
            #    ref_schema == schema and
            #    ref_n_features == n_features and
            #    ref_n_trees == n_trees):

            if (ref_schema == schema and
                ref_n_features == n_features and
                ref_n_trees == n_trees):

                pykhiops_dirname = [name for name in os.listdir(run_dpath) if name.startswith('pykhiops_tmp')][0]
                pykhiops_dirpath = f"{run_dpath}/{pykhiops_dirname}"
                report_fpath = f"{pykhiops_dirpath}/TrainEvaluationReport.xls"

                with open(report_fpath, 'r') as fp:
                    lines = fp.readlines()
                    cg_value = float(lines[22].replace('\n', '').split('\t')[1])

                return cg_value
    print("no experiment match...")
    return

#  = "C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\raw"
def get_info_by_dataset(
    dataset_names,
    root_dir_raw):
    """
    Retrieve dataset info
    However the info is already available in
    "data\\results\\all_sota_models_accuracy.csv"
    """

    from load_data import load_data_from_dir

    dataset_info = {
        'dataset_name' : dataset_names,
        'n_classes'    : [],
        'test_size'    : [],
        'train_size'   : [],
        'min_class_count' : []
    }

    for dataset_name in dataset_names:

        dataset_info = []

        train_df, test_df = load_data_from_dir(dataset_name,
                                               rootdir=root_dir_raw)
        # Train/test set size
        train_size = len(train_df)
        test_size = len(test_df)

        # Class count
        n_classes = len(set(test_df[0].tolist()))

        # Ratio minority class
        count_per_class = { c : 0 for c in set(y_train) }
        for el in y_test:
            count_per_class[el] += 1

        import operator
        min_class_count = sorted(
            count_per_class.items(),
            key=operator.itemgetter(1),
            reverse=False)[0][1]

        dataset_info['min_class_count'].append(min_class_count)
        dataset_info['n_classes'].append(n_classes)
        dataset_info['test_size'].append(test_size)
        dataset_info['train_size'].append(train_size)

    return dataset_info

# ="C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\raw"
# ="C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\interim"
def get_acc_by_dataset(
    dataset_names,
    representations,
    schema,
    n_features,
    n_trees,
    root_dir_raw,
    root_dir_interim):
    """
    Computes accuracy for a given dataset, by a given model parameters (representations, schema)
    Use check_experiments_runs() to verify which experiments parameters are available in the interim data dir
    """

    from sklearn.metrics import accuracy_score
    from load_data import load_data_from_dir
    from load_data import load_y_pred

    accuracy_scores = {}
    for dataset_name in dataset_names:
        _, test_df = load_data_from_dir(dataset_name, rootdir=root_dir_raw)
        y_test = test_df[0].tolist()

        for label_pred_dict in load_y_pred(dataset_name, rootdir=root_dir_interim):

            # label_pred_dict = { params, preds}
            # label_score_dict = { params, scores}
            if not (
                #set(label_pred_dict['params']['representations']) == set(representations) and
                label_pred_dict['params']['schema'] == schema and
                label_pred_dict['params']['n_features'] == n_features and
                label_pred_dict['params']['n_trees'] == n_trees):
                continue

            y_pred = label_pred_dict['preds']
            accuracy_scores[dataset_name] = accuracy_score(y_test, y_pred)
            break

    return accuracy_scores


# ="C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\raw"
# ="C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\interim"

def get_auc_by_dataset(
    dataset_names,
    representations,
    schema,
    n_features,
    n_trees,
    root_dir_raw,
    root_dir_interim):
    """
    Computes average per-class AUC for a given dataset, by a given model parameters (representations, schema)
    Use check_experiments_runs() to verify which experiments parameters are available in the interim data dir
    """

    from sklearn.metrics import accuracy_score
    from load_data import load_data_from_dir
    from load_data import load_y_scores

    auc_scores = {}
    for dataset_name in dataset_names:
        _, test_df = load_data_from_dir(dataset_name, rootdir=root_dir_raw)
        y_test = test_df[0].tolist()

        for label_pred_dict in load_y_scores(dataset_name, rootdir=root_dir_interim):

            #label_pred_dict = { params, preds}
            #label_score_dict = { params, scores}
            if not (
                #set(label_pred_dict['params']['representations']) == set(representations) and
                label_pred_dict['params']['schema'] == schema and
                label_pred_dict['params']['n_features'] == n_features and
                label_pred_dict['params']['n_trees'] == n_trees):
                continue

            y_score = label_pred_dict['scores']
            auc_scores[dataset_name] = avg_auc(y_test, y_score)
            break

    return auc_scores

def get_best_model_info(
    dataset_name,
    schema,
    n_features,
    n_trees,
    root_dir_raw,
    root_dir_interim):
    """
    Computes average per-class AUC for a given dataset, by a given model parameters (representations, schema)
    Use check_experiments_runs() to verify which experiments parameters are available in the interim data dir
    """

    from sklearn.metrics import accuracy_score
    from load_data import load_data_from_dir
    from load_data import load_exp_output
    from load_data import load_all_exp_output

    exps_output, exps_khiops_output = load_all_exp_output(dataset_name,root_dir_interim)
    run_time = 0.
    id_exp = 0
    for exp_output in exps_output:

        if (
            exp_output['params']['schema'] == schema and
            exp_output['params']['n_features'] == n_features and
            exp_output['params']['n_trees'] == n_trees):

            selected_rep = exp_output['params']['representations']
            selected_shema = exp_output['params']['schema']
            nb_loop = exp_output['params']['nb_loop_var_selection']
            nb_informative_features = exps_khiops_output[id_exp]['preparationReport']['summary']['informativeVariables']

        run_time = run_time + exp_output['params']['run_time']          # cumulated time to learn all models
        id_exp = id_exp + 1

    return selected_rep, selected_shema, nb_loop, run_time, nb_informative_features


def get_cg_by_dataset(
    dataset_names,
    param_exp):
    """
    Computes average per-class AUC for a given dataset, by a given model parameters (representations, schema)
    Use check_experiments_runs() to verify which experiments parameters are available in the interim data dir
    """
    cg_scores = {}

    for dataset_name in dataset_names:                                              # for each dataset
        param_exp['dataset_name'] = dataset_name
        cg_scores[dataset_name] =  get_cg(param_exp)

    return cg_scores

def get_score_by_dataset(param_exp, path_bench_parameters, score='accuracy'):
    """
    Wraps over get_auc_by_dataset() and get_acc_by_dataset()
    """

    #
    # param_exp = {
    # 'dataset_name' : dataset_name,
    # 'keep_khiops_files' : param_bench['keep_khiops_files'],
    # 'input_dir' : param_bench['input_dir'],
    # 'output_dir' : param_bench['output_dir'],
    # 'representations' : representations,
    # 'schema' : schema,
    # 'n_features' : n_features,
    # 'test_size' : param_bench['test_size'],
    # 'n_trees' : n_trees
    # }

    from load_data import get_dataset_names
    dataset_names = get_dataset_names(path_bench_parameters)

    root_dir_raw = param_exp['input_dir']
    root_dir_interim = param_exp['output_dir']
    schema = param_exp['schema']
    representations = param_exp['representations']

    # TODO:
    n_features = param_exp['n_features']
    n_trees = param_exp['n_trees']

    if score == 'accuracy':
        if schema == 'argmax':
            return argmax_cg_model_select(dataset_names, representations)
        else:
            return get_acc_by_dataset(dataset_names, representations, schema, n_features, n_trees,root_dir_raw, root_dir_interim)

    elif score == 'auc':
        if schema == 'argmax':
            print("Dynamic schema auc eval needs to be implemented...")
        else:
            return get_auc_by_dataset(dataset_names, representations, schema, n_features, n_trees, root_dir_raw, root_dir_interim)

    elif score == 'cg':
        return get_cg_by_dataset(dataset_names,param_exp)

    else:
        print("Invalid `score` parameter")

def get_experiement_name(param_exp):
    """
    Builds a string that reesents the name of an experiment from a dictionary that contains the experiement parameters
    """

    current_expe_name = "Expe_"

    current_expe_name = "Schema_"

    if param_exp['schema'] == 0:
        current_expe_name += "One_each_"
    elif param_exp['schema'] == 1:
        current_expe_name += "All_in_one_"
    elif param_exp['schema'] == 2:
        current_expe_name += "Both_"
    #elif param_exp['schema'] == 3:
    #    current_expe_name += "Best_one_"

    current_expe_name += "N_features_"
    current_expe_name += str(param_exp['n_features'])
    current_expe_name += "_"

    current_expe_name += "N_trees_"
    current_expe_name += str(param_exp['n_trees'])
    current_expe_name += "_"

    #current_expe_name += "Rep_"
    #current_expe_name += "_".join(str(x).lower() for x in param_exp['representations'])
    #current_expe_name += "_"

    return current_expe_name


def wrapper_detailed_result(param_bench):

    import pandas as pd

    # Load result of benchMarck
    with open(param_bench["results_dir"]+"/scores_all_expe_all_dataset.csv", 'r') as fp:
        snbs_df = pd.read_csv(fp, index_col=0)

    result_df = pd.DataFrame(index=param_bench['dataset_names'], columns=['name_best_model','test_AUC','test_ACC','nb_informative_var','nb_selected_representations','selected_representations','selected_shema','nb_loop_rep_selection','run_time_on_dataset'])		# result dataframe : col = detailed results , row = datasets

    for dataset_name in param_bench['dataset_names']:                   # for each dataset

        for n_features in param_bench['n_features']: 					# for each nb_features
            for n_trees in param_bench['n_trees']:						# for each number of trees
                for schema in param_bench['schema']:

                    param_exp = {
                    'dataset_name' : "",
                    'input_dir' : param_bench['input_dir'],
                    'output_dir' : param_bench['output_dir'],
                    'representations' : param_bench['representations'],
                    'schema' : schema,
                    'n_features' : n_features,
                    'test_size' : param_bench['test_size'],
                    'n_trees' : n_trees
                    }

                    current_expe_name = get_experiement_name(param_exp)

                    if(current_expe_name == snbs_df['Best_CG_Expe'][dataset_name]):

                        # colonnes du dataset de resultats detailles

                        # current_expe_name

                        AUC_best_model = get_auc_by_dataset([dataset_name],
                            param_exp['representations'],
                            param_exp['schema'],
                            param_exp['n_features'],
                            param_exp['n_trees'],
                            param_exp['input_dir'],
                            param_exp['output_dir'])[dataset_name]

                        ACC_best_model = get_acc_by_dataset([dataset_name],
                            param_exp['representations'],
                            param_exp['schema'],
                            param_exp['n_features'],
                            param_exp['n_trees'],
                            param_exp['input_dir'],
                            param_exp['output_dir'])[dataset_name]

                        # -------------------------------------

                        selected_rep, selected_shema, nb_loop, run_time, nb_informative_features = get_best_model_info(
                            dataset_name,
                            param_exp['schema'],
                            param_exp['n_features'],
                            param_exp['n_trees'],
                            param_exp['input_dir'],
                            param_exp['output_dir'])

                        nb_selected_rep = len(selected_rep)

                        # store detailed results
                        result_df['name_best_model'][dataset_name] = current_expe_name
                        result_df['test_AUC'][dataset_name] = AUC_best_model
                        result_df['test_ACC'][dataset_name] = ACC_best_model
                        result_df['nb_informative_var'][dataset_name] = nb_informative_features
                        result_df['nb_selected_representations'][dataset_name] = nb_selected_rep
                        result_df['selected_representations'][dataset_name] = str(selected_rep)
                        result_df['selected_shema'][dataset_name] = selected_shema
                        result_df['nb_loop_rep_selection'][dataset_name] = nb_loop
                        result_df['run_time_on_dataset'][dataset_name] = run_time

    result_df.to_csv(param_bench['results_dir']+"/detailed_results_best_models.csv")



if __name__ == "__main__":

    print('cf. benchmark notebook')
