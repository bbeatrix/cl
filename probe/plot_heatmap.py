import seaborn as sns
import glob
import os
import pickle
import pandas as pd
import re

files = glob.glob('results/**/*.pkl')


records = []

for filepath in files:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        record = {}
        record['front_layer'] = data['params']['front_layer']
        #record['end_layer'] = data['params']['end_layer']
        record['frank_task_idx'] = data['params']['task_idx']

        record['front_acc'] = data['model_results']['front']['acc']

        record['front_acc_on_task'] = data['model_results']['front_on_task']['acc']
        record['end_acc_on_task'] = data['model_results']['end_on_task']['acc']
        record['trans_acc_on_task'] = data['model_results']['trans_on_task']['acc']

        record['end_acc'] = data['model_results']['end']['acc']
        record['trans_acc'] = data['model_results']['trans']['acc']

        record['psinv_acc'] = data['model_results']['ps_inv']['acc']

        record['cka'] = data['cka']
        record['cka_frank'] = data['cka_frank']
        record['l2'] = data['l2']
        record['l2_frank'] = data['l2_frank']
        record['cka_ps_inv'] = data['cka_ps_inv']
        #record['l2_ps_inv'] = data['l2_ps_inv']

        record['ps_inv_rel_acc'] = data['m2_sim']['ps_inv']['rel_acc']
        record['after_rel_acc'] = data['m2_sim_on_task']['after']['rel_acc']
        record['m1_rel_acc'] = data['m2_sim']['m1']['rel_acc']

        front_model = data['params']['front_model']
        end_model = data['params']['end_model']
        
        print(front_model)
        mathches = re.search('num_epochs_(.*)\.', front_model, re.IGNORECASE)
        record['front_model_epochs'] = int(mathches.group(1))
        
        mathches = re.search('task_([0-9]*)_of_([0-9]*)_', front_model, re.IGNORECASE)
        record['front_model_at_task_end'] = mathches.group(1)
        #record['front_model_num_tasks'] = mathches.group(2)
        
        records.append(record) 


df = pd.DataFrame(records)

layer_name = 'conv1'
task_idx = 3
front_model_epochs = 40

print(df)
#filtered = df[(df['front_model_epochs'] == front_model_epochs) & (df['front_layer'] == layer_name) & (df['frank_task_idx'] == task_idx)]
df = df[(df['front_model_epochs'] == front_model_epochs) & (df['frank_task_idx'] == task_idx)]

print(df)

pivoted = df.pivot('front_layer', 'front_model_at_task_end', 'after_rel_acc')
pivoted.rename(columns={'front_model_at_task_end': 'CL model after task #', 'fron_layer': 'Layer', 'after_rel_acc': 'Relative accuracy'})
heatmap_plot = sns.heatmap(pivoted, annot=True, fmt='.2f', linewidth=.5)
heatmap_plot.figure.savefig('heatmap.png')
