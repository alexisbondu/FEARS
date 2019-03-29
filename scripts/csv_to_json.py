
# Prior to running script:
# 1. Go to http://timeseriesclassification.com/dataset.php
# 2. Copy / Paste table to Excel
# 3. Export to .csv (sep = ';' by default)

import pandas as pd
import simplejson as json 

csv_fpath = "C:\\Users\\rdwp8532\\Desktop\\workThat\\script_prettyaf\\dataset_types.csv"
csv_json = "C:\\Users\\rdwp8532\\Desktop\\workThat\\script_prettyaf\\dataset_types.json"

data_df = pd.read_csv(csv_fpath, header=0, sep=';')
data_dict = { k : v[0] for k, v in data_df.set_index('Dataset').T.to_dict('list').items() }

with open(csv_json, 'w') as fp:
	json.dump(data_dict, fp)

