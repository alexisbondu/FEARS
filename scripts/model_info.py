# parse variables du map

def get_map_vars(fpath):
    
	from io import StringIO
	import pandas as pd


	# Read xls file
	with open(fpath, 'r') as fp:
		doc_lines = fp.readlines()

	# Keep generated variables lines
	keep = False
	doc_lines_keep = ['Prepared name\tName\tLevel\tWeight\tMAP\n']
	for line in doc_lines:
		if keep:
			doc_lines_keep.append(line)
		elif line == doc_lines_keep[0]:
			keep = True

	# Convert to str
	doc_lines_keep_str = ''.join(doc_lines_keep)

	# Use StringIO to read as CSV to pandas DataFrame
	data = StringIO(doc_lines_keep_str)
	variables_df = pd.read_csv(data, sep="\t")

	# Get list of MAP variables
	map_vars = variables_df[variables_df['MAP'] == 1.0]['Name'].tolist()

	return map_vars


def get_representation_count(var_name):

	import re

	representation_names = ['TS', 'D', 'DD', 'CUMSUM', 'DCUMSUM', 'ACF', 'PS']
	matches = {r_name : 0 for r_name in representation_names}

	match_map = {"val_{}[ )]".format(r_name) : r_name for r_name in representation_names}

	for representation_re, representation_name in match_map.items():
		match_count = len(re.findall(representation_re, var_name))
		matches[representation_name] += match_count

	return matches


def get_depth_count(var_name):
	"""
	eg:
	var_name = 'Count(additionalDataTable8)  # depth = 1
		where val_DD <= 0.00075965           # depth = 2
		and val_TS in ]-0.927795, -0.09971]' # depth = 3
	"""
	import re

	match_where = len(re.findall(' where ', var_name))
	match_and = len(re.findall(' and ', var_name))

	if match_where < 1:
		return 1
	else:
		return 2 + match_and


def get_dataset_category(dataset_name,
	dataset_type_json_fpath = "C:\\Users\\rdwp8532\\Desktop\\workThat\\script_prettyaf\\dataset_types.json"):

	import simplejson as json

	with open(dataset_type_json_fpath) as fp:
		type_dict = json.load(fp)

	return type_dict[dataset_name]

# import pandas as pd
# representation_count_df = pd.DataFrame(matches, columns=representation_names)

# if binary_count:
#     representation_count_bin_df = representation_count_df.copy()
#     for col in list(representation_count_df):
#         mask = representation_count_df[col] > 0
#         representation_count_bin_df.loc[mask, col] = 1
#     return representation_count_bin_df

if __name__ == "__main__":

	fpath = "C:\\Users\\rdwp8532\\Desktop\\workThat\\data\\interim\\Adiac\\014\\pykhiops_tmp1489574023352\\ModelingReport.xls"
	map_vars = get_map_vars(fpath)

	print(get_dataset_type('Adiac'))