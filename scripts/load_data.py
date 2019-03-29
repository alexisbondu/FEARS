

#  = "/home/brain/Documents/python_workspace/data/raw"

def load_data_from_dir(dataset_name,
    rootdir,
    convert_classes=True, v=False):
    """
    Input abs path of directory containing dataset_TRAIN.txt and dataset_TEST.txt
    as per timeseriesclassification.com
    Returns single pandas df
    """

    def convert_classes(input_df):
        ini_classes = input_df[input_df.columns[0]].values
        class_map = {
            class_ini : i
            for i, class_ini in enumerate(sorted(list(set(ini_classes))))
            }
        mapped_classes = [class_map[class_ini] for class_ini in ini_classes]
        res_df = input_df.copy()
        res_df[0] = mapped_classes
        return res_df

    dpath = "{}/{}".format(rootdir, dataset_name)

    if v:
        print("[+] Loading input data")
        print("    {}".format(dpath))

    import os
    import pandas as pd
    for child in os.listdir(dpath):
        fpath = os.path.join(dpath, child)
        if os.path.isfile(fpath):
            if fpath.endswith('_TRAIN.txt'):
                train_df = pd.read_csv(fpath, header=None)

                if v:
                    print("\t[+] Loaded train data")
                    print("\t    {}".format(fpath))

                if convert_classes:
                    train_df = convert_classes(train_df)

            elif fpath.endswith('_TEST.txt'):
                test_df = pd.read_csv(fpath, header=None)

                if v:
                    print("\t[+] Loaded test data")
                    print("\t    {}".format(fpath))

                if convert_classes:
                    test_df = convert_classes(test_df)

    test_df.index = test_df.index + max(train_df.index) + 1


    return train_df, test_df


#  = "/home/brain/Documents/python_workspace/data/raw"
def load_y_test(dataset_name,
    rootdir):

    _, test_df = load_data_from_dir(dataset_name, rootdir)
    y_test = test_df[0].tolist()

    return y_test


# = "/home/brain/Documents/python_workspace/data/interim"

def load_exp_output(dataset_name,
    rootdir):

    import os
    import simplejson as json

    dataset_exp_dpath = "{}/{}".format(rootdir, dataset_name)

    exp_output = []

    for run_dir in os.listdir(dataset_exp_dpath):
        run_dir_dpath = "{}/{}".format(dataset_exp_dpath, run_dir)

        for fname in os.listdir(run_dir_dpath):
            fpath = "{}/{}".format(run_dir_dpath, fname)

            try:
                with open(fpath, 'r') as fp:
                    output = json.load(fp)
                    exp_output.append(output)
            except Exception as e:
                continue

    return exp_output


def load_all_exp_output(dataset_name,
    rootdir):

    import os
    import simplejson as json

    dataset_exp_dpath = "{}/{}".format(rootdir, dataset_name)

    exp_output = []

    for run_dir in os.listdir(dataset_exp_dpath):
        run_dir_dpath = "{}/{}".format(dataset_exp_dpath, run_dir)

        # find the path of the khiop temp folder
        list_file_and_rep = os.listdir(run_dir_dpath)
        khiops_temp_folder = ""
        for current_name in list_file_and_rep:
            if current_name.find("pykhiops_tmp")==0:
                khiops_temp_folder=current_name
                break

        khiops_folder_path = "{}/{}".format(run_dir_dpath,khiops_temp_folder)

        for fname in os.listdir(khiops_folder_path):
            fpath = "{}/{}".format(khiops_folder_path, fname)

            try:
                with open(fpath, 'r') as fp:
                    output = json.load(fp)
                    exp_output.append(output)
            except Exception as e:
                continue

    return load_exp_output(dataset_name,rootdir), exp_output


# = "/home/brain/Documents/python_workspace/data/interim"

def load_y_pred(dataset_name,
    rootdir):

    import os
    import pandas as pd
    import simplejson as json

    labels_preds = []

    exp_output = load_exp_output(dataset_name, rootdir)

    for output in exp_output:

        # list( (label, score) for each label )
        pred_tuples = [(int(k), v) for k, v in output["results"]["Predictedtarget"].items()]
        pred_tuples_sorted = sorted(pred_tuples, key=lambda x: x[0] )

        # list( score for each label_sorted )
        y_pred = [t[1] for t in pred_tuples_sorted]
        params = output['params']

        labels_preds.append({'params' : params, 'preds' : y_pred})

    return labels_preds



#  = "/home/brain/Documents/python_workspace/data/raw/interim"

def load_y_scores(dataset_name,
    rootdir):

    import os
    import pandas as pd
    import simplejson as json

    from load_data import load_exp_output

    exp_output = load_exp_output(dataset_name, rootdir)

    scores_preds = []
    for output in exp_output:
        output['results'].pop('Predictedtarget')
        scores_dict_raw = output['results']
        scores_dict_clean = {}

        for colname, series_dict in scores_dict_raw.items():
            colname = int(colname.split('Probtarget')[1]) # rename column to int label value
            series_dict = { int(k) :  v for k, v in series_dict.items() }
            scores_dict_clean[colname] = series_dict

        scores = pd.DataFrame.from_dict(scores_dict_clean).values.tolist()
        params = output['params']

        scores_preds.append({'params' : params, 'scores' : scores})

    return scores_preds


# = "/home/brain/Documents/python_workspace/dev/filelist.json"

def get_dataset_names(
    path_bench_parameters_json):

    import simplejson as json
    with open(path_bench_parameters_json, 'r') as file:
        bench_parameters = json.load(file)
    return bench_parameters['dataset_names']

# ======================================================================================
def check_invalid_datasets(data_root, dataset_list=[]):
    """
    Input abs path directory containing dataset dirs
    Tries loading and print log if failed to locate data or data didnt load properly
    """
    import os
    dataset_lookup = dataset_list
    if dataset_lookup == []:
        dataset_lookup = os.listdir(data_root)

    for dname in os.listdir(data_root):
        if len(dataset_list) > 0 and dname not in dataset_list:
            continue
        fpath = os.path.join(data_root, dname)
        try:
            load_data_from_dir(fpath)
        except:
            print("Could not find valid dataset file for dataset: {}".format(fpath))
            if os.path.isdir(fpath):
                print("Files in dataset dir...")
                for item in os.listdir(fpath):
                    print("\t{}".format(item))



if __name__ == "__main__":

    # Check invalid datasets
    #data_root = "C:/Users/rdwp8532/Desktop/workThat/data/raw/"
    #check_invalid_datasets(data_root)

    # Load sample dataset
    dpath = "/home/brain/Documents/python_workspace/data/raw/DiatomSizeReduction"
    train_df, test_df = load_data_from_dir("DiatomSizeReduction")
    print(test_df.head(20))
