from dlhub_sdk.models.servables.sklearn import ScikitLearnModel
from dlhub_sdk.models.datasets import TabularDataset
from dlhub_sdk import DLHubClient

import pandas as pd
import json

# Data used for model
df = pd.read_csv('GB_2_data.csv')
features = df.drop(columns=['composition', 'dmax'], axis=0).columns.to_list()
label = 'dmax'
case = 'composition'

# Data information
data_info = TabularDataset.create_model('GB_2_data.csv', read_kwargs=dict(header=1))
data_info.set_domains(['material science', 'metallic glasses'])
data_info.load_dataset('GB_2_data.csv', 'csv')

# Annotete columns
for i in df.columns:
    if i == case:
        data_info.annotate_column(i, description='The material composition')
    elif i == label:
        data_info.annotate_column(i, description='The casting diameter for glassy metals', units='mm')
    else:
        data_info.annotate_column(i, description='Generated arithmetic feature')

# Specify inputs and outputs of model
data_info.mark_inputs(features)
data_info.mark_labels([label])

# Describe provenance
data_info.set_title('Data for fitting cirtical casting diamter to common characteristic temperatures.')
data_info.set_name('metallic_glasses_GB_2_data')
data_info.set_authors(["Schultz, Lane"])

# Sklearn model
model_info = ScikitLearnModel.create_model(
                                           'GB_2_model.pickle',
                                           n_input_columns=len(features)
                                           )

# Model descrition
model_info.set_title('Model for Predicting Critical Casting Diameters from Characteristic Temperatures')
model_info.set_name('GB_2')
model_info.set_domains(['material science', 'metallic glasses'])

# Print out the result
print('--> Dataset Information <--')
print(json.dumps(data_info.to_dict(), indent=2))
print('\n--> Model Information <--')
print(json.dumps(model_info.to_dict(), indent=2))
