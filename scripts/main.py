
# TODO sur la VM
# - conda install simplejson
# - conda install orange3

def make_dir(dpath):
	import os, errno
	try:
		os.makedirs(dpath)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


def get_next_run_nb(dpath):
	import os
	dnames = os.listdir(dpath)
	run_nbs = []
	for dname in dnames:
		try:
			run_nbs.append(int(dname))
		except:
			continue
	try:
		new_run_nbs = max(run_nbs) + 1
	except:
		new_run_nbs = 0
	return f"{new_run_nbs:03}"


def experiment_wrapper(params_exp, test_eval=True, nb_sample=None):
	"""
	param_exp = {
		'input_dir' : "/home/brain/Documents/python_workspace/data/raw",
		'output_dir' : "/home/brain/Documents/python_workspace/data/interim",
		'keep_khiops_files' : True
		'dataset_name' : None,
		'representations' : ['TS', 'D', 'DD', 'CUMSUM', 'DCUMSUM', 'ACF', 'PS'],
		'schema' : 1,
		'n_features' : 20000,
		'test_size' : None,
		'n_trees' : 0    # New : trigger additional generated feature by using MODL-trees in the root table
	}

	schema:
		0 : one table per representation
		1 : all representation in a single table
		2 : combination of `0` and `1`

	n_trees:
		If positive triggers additional features that comes from MODL random trees computed from the main table
	"""
	import time
	start_time = time.time()
	# Parse inputs
	input_dir = params_exp['input_dir']
	output_dir = params_exp['output_dir']
	dataset_name = params_exp['dataset_name']
	representations = params_exp['representations']
	schema = params_exp['schema']
	n_features = params_exp['n_features']
	test_size  = params_exp['test_size']
	n_trees = params_exp['n_trees']

	# Input/output path setting
	output_fpath = "{}/{}".format(output_dir, dataset_name)

	# Create directiory for dataset experiments (if necessary)
	dataset_output_dpath = "{}/{}".format(output_dir, dataset_name)
	make_dir(dataset_output_dpath)

	# Create dir for experiment run (always)
	run_nb = get_next_run_nb(dataset_output_dpath)###########################################  run_nb
	run_output_dpath = "{}/{}".format(dataset_output_dpath, run_nb)
	make_dir(run_output_dpath)

	# Prepare input data
	kw_get_inputs = {
		"dataset_name" : dataset_name,
		"data_root" : input_dir,
		"representations" : representations,
		"schema" : schema,
		"test_size" : test_size,
		"nb_sample" : nb_sample
		}

	#dataset_name = param_exp['dataset_name']
    #representations = set(param_exp['representations'])
    #schema = param_exp['schema']

	additional_tables_train, additional_tables_test, train_ids, test_ids, y_train, y_test = get_inputs(**kw_get_inputs)

	# Model training
	#print("[+] Classifier training")

	clt = PyKhiopsClassifier(auto_clean=False, computation_dir=run_output_dpath)
	clt.fit(X_list=additional_tables_train, y=y_train, n_features=n_features, n_trees=n_trees)
	end_time = time.time()

	# Predictions
	#print("[+] Classifier prediction")

	if test_eval:
		pred_test = clt.predict_proba(X=test_ids, X_list=additional_tables_test)
		# Accuracy
		y_test_pred = pred_test['Predictedtarget'].values
		acc = accuracy_score(y_test_pred, y_test)
	else:
		pred_test = pd.DataFrame()
		y_test_pred = pd.Series()
		acc = 0.

	#print("    accuracy score: {}".format(round(acc, 2)))

	# Netoyage run_output_dpath (not keep_khiops_files)

	# find the name of the khiops temporary folder
	list_file_and_rep = os.listdir(run_output_dpath)
	khiops_temp_folder = ""
	for current_name in list_file_and_rep:
		if current_name.find("pykhiops_tmp")==0:
			khiops_temp_folder=current_name
			break

	# protected files
	protected_files = ['TrainEvaluationReport.xls','AllReports.json','Modeling.kdic','ModelingReport.xls','PreparationReport.xls','TrainEvaluationReport.xls']

	# remove file, except TrainEvaluationReport.xls
	list_file_and_rep = os.listdir(run_output_dpath+"/"+khiops_temp_folder)
	for current_name in list_file_and_rep:
		if current_name not in protected_files:
			os.remove(run_output_dpath+"/"+khiops_temp_folder+"/"+current_name)

	# Save performances and the parameters
	params_json = {
		'representations' : representations,
		'schema' : schema,
		'n_features' : n_features,
		'n_trees' : n_trees
		}

	if 'nb_loop_var_selection' in params_exp:
		params_json['nb_loop_var_selection'] = params_exp['nb_loop_var_selection']

	if 'run_time' in params_exp:
		params_json['run_time'] = params_exp['run_time'] + (end_time - start_time)

	output_data = {
		'dataset_name' : dataset_name,
		'params' : params_json,
		'accuracy' : acc,
		#'compression_gain' : CG,
		'results' : pred_test.to_dict(),
		'preds' : y_test_pred.tolist()
		}

	#print("[+] Saving results")

	# TODO : un suffix plus explicite avec la valeur des params

	ts_suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	output_fpath = "{}/{}.json".format(run_output_dpath, ts_suffix)
	with open(output_fpath, 'w') as fp:
		json.dump(output_data, fp, indent=2)
	#print("    {}".format(output_fpath))

	# Compression gain
	kw_get_inputs['n_features'] = n_features
	kw_get_inputs['n_trees'] = n_trees
	kw_get_inputs['output_dir'] = output_dir
	CG = get_cg(kw_get_inputs)

	return CG


##################################################################################
#                 Schema and representation selection
##################################################################################

def find_best_schema(param_bench, param_exp):
	"""
	find the best schema on limited data
	"""

	CG_by_Schema = {}

	for schema in param_bench['schema']: 			  					# for each Shema

		# building the parameters of the current exeperiment
		param_exp['schema'] = schema

		# learning step
		CG = experiment_wrapper(param_exp, test_eval=False)
		CG_by_Schema[schema] = CG

	# the best schema with the most important compression gain
	inverse_CG_by_Schema = [(value, key) for key, value in CG_by_Schema.items()]
	best_schema = max(inverse_CG_by_Schema)[1]
	all_rep_CG = max(inverse_CG_by_Schema)[0]

	# cleaning the khiops prov files
	if os.path.isdir(param_exp['output_dir']):
		shutil.rmtree(param_exp['output_dir'])

	return best_schema, all_rep_CG

##################################################################################

def feedforward(shuffle_rep, selected_rep, best_CG, param_exp, stop_rep_selection, nb_sample, delta_move):
	"""
	feedforward step of the representation selection algorithm
	"""
	# feed forward
	for current_rep in shuffle_rep:
		if current_rep not in selected_rep:
			param_exp['representations'] = selected_rep+[current_rep]
			current_CG = experiment_wrapper(param_exp, test_eval=False, nb_sample=nb_sample)

			if current_CG > (best_CG + delta_move):
				best_CG = current_CG
				selected_rep = selected_rep+[current_rep]
				stop_rep_selection = False

	# shuffling the list of the representations
	shuffle(shuffle_rep)

	# cleaning the khiops prov files
	if os.path.isdir(param_exp['output_dir']):
		shutil.rmtree(param_exp['output_dir'])

	return shuffle_rep, selected_rep, best_CG, stop_rep_selection

##################################################################################

def feedbackward(shuffle_rep, selected_rep, best_CG, param_exp, stop_rep_selection, nb_sample, delta_move):
	"""
	feedbackward step of the representation selection algorithm
	"""
	# feed backward
	for current_rep in shuffle_rep:
		if current_rep in selected_rep and len(selected_rep) > 1:

			prov_selected_rep = copy.deepcopy(selected_rep)
			prov_selected_rep.remove(current_rep)
			param_exp['representations'] = prov_selected_rep
			current_CG = experiment_wrapper(param_exp, test_eval=False, nb_sample=nb_sample)

			if current_CG > (best_CG + delta_move):
				best_CG = current_CG
				selected_rep = prov_selected_rep
				stop_rep_selection = False

	# shuffling the list of the representations
	shuffle(shuffle_rep)

	# cleaning the khiops prov files
	if os.path.isdir(param_exp['output_dir']):
		shutil.rmtree(param_exp['output_dir'])

	return shuffle_rep, selected_rep, best_CG, stop_rep_selection


###################################################################################

def rep_selection_and_learning(args):
	import time
	param_bench, dataset_name = args

	# wait while the RAM memory is almost full
	ram_use = psutil.virtual_memory()

	while ram_use.percent > 80:
		time.sleep(30)
		ram_use = psutil.virtual_memory()


	for n_trees in param_bench['n_trees']:									# for each number of trees
		for schema in param_bench['schema']:								# for each shema

			# representation selection
			# ------------------------

			param_exp = {
			'dataset_name' : dataset_name,
			'input_dir' : param_bench['input_dir'],
			'output_dir' : param_bench['output_dir']+"/prov_"+dataset_name,
			'representations' : param_bench['representations'],
			'n_features' : param_bench['nb_features_for_rep_selection'],
			'test_size' : param_bench['test_size'],
			'n_trees' : n_trees,
			'schema' : schema
			}

			# initialisation
			start_time = time.time()
			shuffle_rep = param_exp['representations']
			selected_rep = shuffle_rep
			param_exp['representations'] = selected_rep
			best_CG = experiment_wrapper(param_exp, test_eval=False)

			#print("+++++ "+str(param_exp['representations'])+"    CG : "+str(best_CG))

			# if the starting point algo is triggered
			if param_bench['trigger_starting_point']:
				#print("++++ searching for best starting point")
				# best starting point
				other_starting_points = param_bench['starting_points']

				for starting_point in other_starting_points:
					param_exp['representations'] = starting_point
					current_CG = experiment_wrapper(param_exp, test_eval=False)

					#print("+++++ "+str(param_exp['representations'])+"    CG : "+str(current_CG))

					if current_CG > best_CG:
						best_CG = current_CG
						selected_rep = starting_point
					# cleaning the khiops prov files
					if os.path.isdir(param_exp['output_dir']):
						shutil.rmtree(param_exp['output_dir'])

			# if the feedforward/backward algo is triggered
			if param_bench['trigger_feedforward']:

				stop_rep_selection = False
				nb_loop = 1

				# feedforward / feedbackward algorithm (stop when the selected sub-set of representation is stable)
				while not stop_rep_selection :

					stop_rep_selection = True
					nb_loop = nb_loop + 1
					# feedbackward
					shuffle_rep, selected_rep, best_CG, stop_rep_selection = feedbackward(shuffle_rep, selected_rep, best_CG, param_exp, stop_rep_selection, nb_sample_for_rep_selection, param_bench["delta_move"])

					# feedforward
					shuffle_rep, selected_rep, best_CG, stop_rep_selection = feedforward(shuffle_rep, selected_rep, best_CG, param_exp, stop_rep_selection, nb_sample_for_rep_selection, param_bench["delta_move"])

			# Update parameters
			end_time = time.time()
			param_exp['representations'] = selected_rep
			param_exp['output_dir'] = param_bench['output_dir']
			param_exp['nb_loop_var_selection'] = nb_loop
			param_exp['run_time'] = (end_time - start_time)

			print(str(datetime.datetime.now())+"\t"+dataset_name+"\t"+str(selected_rep)+"\t"+str(schema)+"\t"+str(best_CG))



			# final learning stage
			for n_features in param_bench['n_features']: 								# for each nb_feature
				param_exp['n_features'] = n_features

				experiment_wrapper(param_exp, test_eval=True)

##################################################################################

def rep_selection_vs_single_rep(args):
	import time
	param_bench, dataset_name, selected_rep_string, selected_shema = args

	# wait while the RAM memory is almost full
	ram_use = psutil.virtual_memory()

	while ram_use.percent > 80:
		time.sleep(30)
		ram_use = psutil.virtual_memory()

	selected_rep = selected_rep_string.replace("[","").replace("]","").replace("'","").replace(" ","").split(",")

	best_CG = 0.0
	best_single_rep = ""

	for single_rep in selected_rep:

		# representation selection
		# ------------------------

		param_exp = {
		'dataset_name' : dataset_name,
		'input_dir' : param_bench['input_dir'],
		'output_dir' : param_bench['output_dir']+"/single_rep_expe/prov_"+dataset_name,
		'representations' : [single_rep],
		'n_features' : param_bench['nb_features_for_rep_selection'],
		'test_size' : param_bench['test_size'],
		'n_trees' : 0,
		'schema' : int(selected_shema)
		}

		current_CG = experiment_wrapper(param_exp, test_eval=False)

		if current_CG > best_CG:
			best_CG = current_CG
			best_single_rep = single_rep

		# cleaning the khiops prov files
		if os.path.isdir(param_exp['output_dir']):
			shutil.rmtree(param_exp['output_dir'])

	# # Final learning

	param_exp = {
	'dataset_name' : dataset_name,
	'input_dir' : param_bench['input_dir'],
	'output_dir' : param_bench['output_dir']+"/single_rep_expe/",
	'representations' : [best_single_rep],
	'n_features' : param_bench['nb_features_for_rep_selection'],
	'test_size' : param_bench['test_size'],
	'n_trees' : 0,
	'schema' : int(selected_shema)
	}

	experiment_wrapper(param_exp, test_eval=True)





if __name__ == "__main__":





# TODO list

# - OPTIMISATION : eviter de lire / echantilloner / préparer les donnees, calculer les representtaions a chaque iteration du feed forward backward !!!!
# - DEBUG : dans prepare_data.py / get_all_in_one_table, gérer le fait que time_representations peut etre vide !!
# - IDEES A TESTER pour ameliorer l'algo de selection de representations :
#        - multi-start
#        - backward puis foreward
#        - Depart avec un sous-ensemble aleatoire de representations
#        - Augmenter progressivement N et f ?

	#import and Khiops environement setting
	import os
	import sys
	import shutil
	from load_data import get_dataset_names
	os.environ["KHIOPSHOME"] = "/opt/khiops" # add KHIOPSHOME to env variables
	sys.path.append("/home/brain/pykhiops/lib") # add pykhiops to PYTHONPATH

	from prepare_data import get_inputs
	from pykhiops.sklearnwrapper import PyKhiopsClassifier
	from sklearn.metrics import accuracy_score
	import simplejson as json
	import datetime
	import multiprocessing
	from multiprocessing import Process
	import copy
	import psutil
	import time

	from analyze_res import get_info_by_dataset
	from analyze_res import get_score_by_dataset
	from analyze_res import get_experiement_name
	from analyze_res import get_cg
	from analyze_res import wrapper_detailed_result

	from nemenyi import wrapper_nemenyi_test
	from nemenyi import wrapper_ranks_vs_Ntrain
	from nemenyi import wrapper_delaited_results
	from nemenyi import wrapper_use_of_rep_plot
	from nemenyi import wrapper_acc_single_rep_plot
	from nemenyi import wrapper_running_time_detailed_plot

	import pandas as pd

	from functools import reduce

	from random import shuffle

	# I - import benchmark parameters from a json file


	#path_bench_parameters = "/home/brain/Documents/python_workspace/dev/bench_param_A.json"
	path_bench_parameters = sys.argv[1]

	with open(path_bench_parameters) as file:
		param_bench = json.load(file)


	# # II - Benchmark loop
	# # -------------------

	scores_df = pd.DataFrame(index=param_bench['dataset_names'])					# scores dataframe : col = exeperiments , row = datasets
	compression_gain_df = pd.DataFrame(index=param_bench['dataset_names'])			# cg dataframe : col = exeperiments , row = datasets
	nb_sample_for_rep_selection = param_bench['nb_sample_for_rep_selection'] 		# the representation selection is carried out by using a limitted rows number

	if param_bench["run_learning_stage"]:

		nb_core = multiprocessing.cpu_count() -1										# count the number of available cores -1

		# multi thead

		pool = multiprocessing.Pool(nb_core)

		func_args = []
		for dataset_name in param_bench['dataset_names']:
			func_args.append((param_bench,dataset_name,))

		try:
			pool.map_async(rep_selection_and_learning, func_args).get(9999999)
		except KeyboardInterrupt:
			# Allow ^C to interrupt from any thread.
			sys.stdout.write('\033[0m')
			sys.stdout.write('User Interupt\n')

		pool.close()

		# III - saving results of experiment
		# -----------------------------------

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

					# getting the accuracy for all dataset given the current parameters
					current_accuracy_by_dataset = get_score_by_dataset(
					    param_exp =param_exp,
						path_bench_parameters = path_bench_parameters,
					    score='accuracy')

					# getting the cg for all dataset given the current parameters
					current_cg_by_dataset = get_score_by_dataset(
					    param_exp =param_exp,
						path_bench_parameters = path_bench_parameters,
					    score='cg')

					# compute the name of the current experiement
					current_expe_name = get_experiement_name(param_exp)

					# storing the new scores into the score_df and cg dataframe
					scores_df = pd.concat([scores_df,pd.DataFrame.from_dict(current_accuracy_by_dataset, orient='index', columns=[current_expe_name])], axis=1)
					compression_gain_df = pd.concat([compression_gain_df,pd.DataFrame.from_dict(current_cg_by_dataset, orient='index', columns=[current_expe_name])], axis=1)

		# best_expe_on_level
		scores_df['Best_CG_Expe'] = compression_gain_df.idxmax(axis=1)
		scores_df['Best_CG_Score'] = scores_df.apply(lambda row: row[row['Best_CG_Expe']], axis=1)

		# save the scores_df
		scores_df.to_csv(param_bench['results_dir']+"/scores_all_expe_all_dataset.csv")
		scores_df.drop(['Best_CG_Expe'],axis=1, inplace=True)

	#---------------------
	#   result analysis
	#---------------------

	if param_bench["run_results_analysis"]:

	# TODO : faire les if qui correspondent au run learning stage du json en entree
	# TODO : Nemenyi test ....
	
		wrapper_detailed_result(param_bench)
		wrapper_nemenyi_test(param_bench)
		wrapper_ranks_vs_Ntrain(param_bench)
		wrapper_delaited_result_plot(param_bench)
		wrapper_use_of_rep_plot(param_bench)
		wrapper_acc_single_rep_plot(param_bench)
		wrapper_running_time_detailed_plot(param_bench)

	#-------------------------------------------------------------------
	# additional expe comparing the single best rep to the selected rep
	#-------------------------------------------------------------------

	if param_bench["run_additional_single_rep_expe"]:

		with open(param_bench["results_dir"]+"/detailed_results_best_models.csv", 'r') as fp:
			detailed_results_df = pd.read_csv(fp, index_col=0)

		nb_core = multiprocessing.cpu_count() -1										# count the number of available cores -1
		pool = multiprocessing.Pool(nb_core)

		func_args = []
		for dataset_name in param_bench['dataset_names']:
			selected_rep = detailed_results_df["selected_representations"][dataset_name]
			selected_shema = detailed_results_df["selected_shema"][dataset_name]
			func_args.append((param_bench,dataset_name,selected_rep,selected_shema,))

		try:
			pool.map_async(rep_selection_vs_single_rep, func_args).get(9999999)
		except KeyboardInterrupt:
			# Allow ^C to interrupt from any thread.
			sys.stdout.write('\033[0m')
			sys.stdout.write('User Interupt\n')

		pool.close()


	# End of script ---------------------

	if param_bench["run_learning_stage"]:
		# zipper les fichiers d'interim
		path_archive = "/".join(param_bench["results_dir"].split("/")[:-1])+"/"+param_bench["output_dir"].split("/")[-1]+".zip"
		os.system("zip -r "+path_archive+" "+param_bench["output_dir"])

		# supprimer les fichiers d'interim
		os.system("rm -f -R "+param_bench["output_dir"]+"/*")
