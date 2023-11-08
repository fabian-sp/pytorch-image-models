from stepback.utils import merge_subfolder

import os
import json
#%% Step 1: merge single files into one 
merge_subfolder(folder_name='../json/vit_tiny_patch16_224', fname='vit_tiny_patch16_224', output_dir='output/clean/')

#%% Step 2: remove unnecessary config keys

output_dir = 'output/clean/'
file_name = 'vit_tiny_patch16_224'

# load merged json
with open(os.path.join(output_dir, file_name) + '.json', "r") as f:
    d = json.load(f)

assert len(d) == 24, "Expected 24 results."

for this in d:
    print(this["config"])
    this["config"]["opt"].pop('opt', None)
    this["config"]["opt"].pop('momentum', None)

# store edited json
with open(os.path.join(output_dir, file_name) + '.json', "w") as f:
    json.dump(d, f, indent=4, sort_keys=True)
    
