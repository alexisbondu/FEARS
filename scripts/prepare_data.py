
import numpy as np
import pandas as pd

def get_rep_df(input_df, rep='TS', index_levels=None):
    """
    Inputs:
        . raw_df, raw TS data as pandas df
        . rep, str for specifying the TS representation
    Output df with a columns cid, pos, value where:
        . cid is index of TS
        . pos is index of time stamp
        . value is value of specified representation for TS `cid` at time `pos`
    """

    from representations import transform_dict

    # Input validation
    assert index_levels in [None, 0, 1, 2],\
    "Invalid `index_levels` value ({})\nShould be in [None, 0, 1, 2]".format(index_levels)

    rep = rep.upper()
    valid_reps = list(transform_dict.keys())
    assert rep in valid_reps,\
    "Invalid `rep` argument value: {rep}\nvalue should be in {valid_reps_str}".format(
        rep=rep, valid_reps_str=','.join(valid_reps))

    # Convert df to np.array; keep only TS data
    ts_df = input_df.drop([0], axis=1) # keep only TS data (drop labels)
    ts_ar = ts_df.values # convert to np.array

    # Save original TS index for later reconstruction
    ts_index = input_df.index

    # Representation transform
    # print("\t[+] .{}".format(rep))
    if rep == 'TS':
        rep_ar = ts_ar # 'TS': keep representation unchanged
    else:
        f_rep = transform_dict[rep.upper()] # get representation transform fct
        rep_ar = f_rep(ts_ar) # apply transform array-wise

    # Flatten data table
    #    before: <cid, t1, t2,...tn>
    #    after: <cid, pos, val>
    n_row, n_col = rep_ar.shape
    cid_data = np.repeat(ts_index, n_col) # repeat each cid for each pos
    pos_data = np.tile(np.arange(n_col), n_row) # repeat each seq of pos for each TS
    val_data = rep_ar.flatten() # flatten values table (single col)

    # Load cols to dict, load dict to df
    col_name = "val_{}".format(rep)
    if not index_levels:
        rep_df_data = {
            'cid' : cid_data,
            'pos' : pos_data,
            col_name : val_data
        }
        rep_df = pd.DataFrame(data=rep_df_data)

    elif index_levels == 1:
        rep_df_data = {
            'pos' : pos_data,
            col_name : val_data
        }
        rep_df = pd.DataFrame(data=rep_df_data, index=cid_data)
        rep_df.index.name = 'cid'

    elif index_levels == 2:
        rep_df_data = {
            'val_pos' : pos_data,
            col_name : val_data
        }
        rep_df = pd.DataFrame(data=rep_df_data, index=[cid_data, pos_data])
        rep_df.index.names = ['cid', 'pos']

    return rep_df


def get_y(input_df):
    """
    Input df as per raw TS data
    Output df with column `target` (used as main table by khiops wrapper)
        and indexed by `cid`
    """
    res_df = pd.DataFrame(
        data={'target' : input_df[0]},
        index=input_df.index
    )
    res_df.index.name = 'cid'
    return res_df


def get_additional_tables(ts_df, representations, schema):
    """
    ts_df: row representation of TS
    representations: list of representation names (check valid acronyms in get_rep_df())
    schema:
        0 : one table per representation
        1 : all representation in a single table
        2 : combination of `0` and `1`
    """
    def get_representation_tables(time_representations):
        representation_tables = [
            get_rep_df(ts_df, rep=rep_name, index_levels=2)
            for rep_name in time_representations ]
        return representation_tables

    def get_all_in_one_table(time_representations):
        import pandas as pd

        # TODO : a debuger !
        if len(time_representations) == 0:
            time_representations = ['TS']

        additional_tables = get_representation_tables(time_representations)
        pos_df = additional_tables[0]['val_pos'].copy()   #### BUG index O ?

        for table_df in additional_tables:
            table_df.drop('val_pos', 1, inplace=True)
        additional_tables.append(pos_df)
        return pd.concat(additional_tables, axis=1, join='outer')
    #--

    if schema == 0:
        representation_tables = get_representation_tables(representations)

    elif schema == 1: # exclude frequency representations
        time_representations = [r for r in representations if r != 'PS']
        all_in_one = get_all_in_one_table(time_representations)
        representation_tables = [all_in_one]

        # add PS back if necessary
        if 'PS' in representations:
            table_PS = get_representation_tables(['PS'])[0]
            representation_tables.append(table_PS)

    elif schema == 2:
        representation_tables = get_representation_tables(representations)
        time_representations = [r for r in representations if r != 'PS']
        all_in_one = get_all_in_one_table(time_representations)
        representation_tables.append(all_in_one)
    else:
        raise ValueError("Invalid schema: {}".format(schema))

    return representation_tables



def get_inputs(
    dataset_name,
    data_root,
    representations,
    schema,
    test_size,
    shuffle=False,
    nb_sample=None):
    """
    prepare the Train and Test sets including secondary tables
    """

    from load_data import load_data_from_dir
    train_df, test_df = load_data_from_dir(dataset_name, data_root) # load entire dataset


    # Sampling the training dataset

    if nb_sample != None:
        if train_df.shape[0] > nb_sample:
            train_df = train_df.sample(n=nb_sample)        

    if test_size is not None:
        from sklearn.model_selection import train_test_split
        print("appending train + test")
        input_df = train_df.append(test_df, ignore_index=True)
        print("splitting train/test")
        train_df, test_df = train_test_split(input_df, test_size=test_size, shuffle=shuffle)

    y_train, y_test = get_y(train_df), get_y(test_df) # extract labels

    train_ids = pd.DataFrame(index = train_df.index) # extract ids of train samples
    test_ids = pd.DataFrame(index = test_df.index) # extract ids of test samples

    #print("[+] Generating representations for TRAIN")
    additional_tables_train = get_additional_tables(train_df, representations, schema)

    #print("[+] Generating representations for TEST")
    additional_tables_test = get_additional_tables(test_df, representations, schema)

    return additional_tables_train, additional_tables_test, train_ids, test_ids, y_train, y_test
