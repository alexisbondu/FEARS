def get_summary_df(
    dataset_names,
    params,
    root_dir_raw,
    root_dir_interim):
    
    from load_data import load_data_from_dir
    from load_data import load_y_pred
    from load_data import load_y_scores
    from load_data import load_exp_output
    
    from analyze_res import avg_auc
    from sklearn.metrics import accuracy_score

    accuracy_score_data = {
        'dataset_name' : dataset_names,
        'accuracy'     : [],
        'avg_auc'      : [],
        'n_classes'    : [],
        'test_size'    : [],
        'train_size'   : [],
        'min_class_count' : []
    }

    for dataset_name in dataset_names:

        accs = []
        aucs = []

        train_df, test_df = load_data_from_dir(dataset_name,
                                               rootdir=root_dir_raw)
        train_size = len(train_df)
        test_size = len(test_df)
        
        y_test = test_df[0].tolist()
        y_train = train_df[0].tolist()

        # Compute aucs, accs 
        for label_pred_dict, label_score_dict in zip(
            load_y_pred(dataset_name, rootdir=root_dir_interim), 
            load_y_scores(dataset_name, rootdir=root_dir_interim) ):
            
            # label_pred_dict = { params, preds}
            # label_score_dict = { params, scores}
            if not (
                set(label_pred_dict['params']['representations']) == set(params['representations']) and
                label_pred_dict['params']['schema'] == params['schema']
            ):
                #print(set(label_pred_dict['params']['representations']))
                #print(label_pred_dict['params']['schema'])
                continue
            
            y_pred = label_pred_dict['preds']
            y_score = label_score_dict['scores']
            

            accs.append( accuracy_score(y_test, y_pred) )
            aucs.append( avg_auc(y_test, y_score))

        # Ratio class majoritaire
        count_per_class = { c : 0 for c in set(y_train) }
        for el in y_test:
            count_per_class[el] += 1

        import operator
        min_class_count = sorted(
            count_per_class.items(),
            key=operator.itemgetter(1),
            reverse=False)[0][1]

        # Fill in result data
        accuracy_score_data['accuracy'].append(round(max(accs), 4))
        accuracy_score_data['avg_auc'].append(round(max(aucs), 4))

        accuracy_score_data['min_class_count'].append(min_class_count)
        accuracy_score_data['n_classes'].append(len(set(y_test)))
        accuracy_score_data['test_size'].append(test_size)
        accuracy_score_data['train_size'].append(train_size)
        
    import pandas as pd
    result_df = pd.DataFrame(data = accuracy_score_data).sort_values(by='accuracy', ascending=False)
    
    return result_df


def load_dataset_names():

    import simplejson as json
    dataset_names_json = "C:\\Users\\rdwp8532\\Desktop\\workThat\\script_prettyaf\\filelist.json"
    with open(dataset_names_json, 'r') as fp:
        dataset_names = json.load(fp)
    return dataset_names



if __name__ == "__main__":

    
    root_dir_raw = "C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\raw"
    root_dir_interim = "C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\interim"

    import pandas as pd
    dataset_names = load_dataset_names()
    result_df = pd.DataFrame(data = {'dataset_name' : dataset_names})

    # Set representation -1 list of list
    all_representations = ['TS', 'D', 'DD', 'CUMSUM', 'DCUMSUM', 'PS', 'ACF']
    for rep_rm in all_representations:
        print(f"MINUS {rep_rm}")
        all_reps_minus_one = [rep for rep in all_representations if rep != rep_rm]
        
        params = {
            'representations' : all_reps_minus_one,
            'schema' : 1
        }

        new_result_df = get_summary_df(dataset_names, params, root_dir_raw, root_dir_interim)
        new_result_df = new_result_df[['dataset_name', 'accuracy', 'avg_auc']]
        new_result_df.columns = ['dataset_name', f'acc_minus_{rep_rm}', f'auc_minus_{rep_rm}']

        result_df = result_df.merge(new_result_df, on='dataset_name', how='outer')

    result_df.to_csv("C:\\Users\\rdwp8532\\Desktop\\representations_benchmark.csv")