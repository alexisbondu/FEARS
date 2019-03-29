
def check_experiments_runs():
	"""
	prints params for exp that have ran on all datasets
	"""
	import os
	import simplejson as json
	from load_data import get_dataset_names

	# List all configs available in exp results per dataset
	data_dpath = "C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\interim"

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


def get_cg(report_fpath):

	with open(report_fpath, 'r') as fp:
		lines = fp.readlines()
		cg_value = float(lines[22].replace('\n', '').split('\t')[1])
		return cg_value


def argmax_cg_model_select(dataset_names, representations):
	"""
	eg: representations=['TS', 'D', 'DD', 'CUMSUM', 'DCUMSUM', 'ACF']
	"""

	import os
	import simplejson as json
	import pandas as pd
	from load_data import get_dataset_names

	scores = {}

	params_set_0 = {
		'representations' : set(representations),
		'schema' : 0,
		'n_features' : 20000
	}

	params_set_1 = {
		'representations' : set(representations),
		'schema' : 1,
		'n_features' : 20000
	}

	data_dpath = "C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\interim"

	for dname in dataset_names:
		dpath_interim = f"{data_dpath}\\{dname}"

		for run_dname in os.listdir(dpath_interim):
			run_dpath = f"{dpath_interim}\\{run_dname}"

			for fname in os.listdir(run_dpath):
				if fname.endswith('.json'):
					json_fpath = f"{run_dpath}\\{fname}"
					with open(json_fpath, 'r') as fp:
						json_content = json.load(fp)

					params = json_content['params']
					params['representations'] = set(json_content['params']['representations'])

					if params == params_set_0 or params == params_set_1:
						pykhiops_dirname = [name for name in os.listdir(run_dpath) if name.startswith('pykhiops_tmp')][0]
						pykhiops_dirpath = f"{run_dpath}\\{pykhiops_dirname}"
						TrainEvaluationReport_fpath = f"{pykhiops_dirpath}\\TrainEvaluationReport.xls"
						CG = get_cg(TrainEvaluationReport_fpath)

						if params == params_set_0:
							CG_0 = CG
							params_set_0_dpath = run_dpath
							results_0 = json_content['results']
						elif params == params_set_1:
							CG_1 = CG
							params_set_1_dpath = run_dpath
							results_1 = json_content['results']

		if CG_0 < CG_1:
			best_params = params_set_1
			best_run_dpath = params_set_1_dpath
			best_labs_pred = pd.DataFrame.from_dict(results_1)['Predictedtarget'].tolist()
		else:
			best_params = params_set_0
			best_run_dpath = params_set_0_dpath
			best_labs_pred = pd.DataFrame.from_dict(results_0)['Predictedtarget'].tolist()

		# Compute accuracy score of model that has best train CG
		# Read actual labels
		from load_data import load_data_from_dir
		_, test_df = load_data_from_dir(dname)
		labs_actual = test_df[0].values
		## Compute accuracy
		from sklearn.metrics import accuracy_score
		acc_score = accuracy_score(best_labs_pred, labs_actual)

		scores[dname] = acc_score

	return scores


if __name__ == "__main__":

	#argmax_cg_model_select()
	check_experiments_runs()


		