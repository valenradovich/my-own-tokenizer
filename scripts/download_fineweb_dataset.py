"""download fineweb edu dataset and filter it to 'x' samples"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset, Dataset

amount_sample = 500_000

dataset = load_dataset("HuggingFaceFW/fineweb-edu", "CC-MAIN-2013-20", split='train', streaming=True)

dataset_sample = []

for i, example in enumerate(dataset):
    dataset_sample.append(example['text'])
    if i == amount_sample:
        break
    
print('samples filtered:', len(dataset_sample))

# Convert to Dataset
filtered_dataset = Dataset.from_dict({'text': dataset_sample})
print('dataset filtered done:', len(filtered_dataset))  

# Save and load the filtered dataset locally
filtered_dataset.save_to_disk(f'data/fineweb_edu_sample_500k')
loaded_filtered_dataset = Dataset.load_from_disk(f'data/fineweb_edu_sample_500k')
print(loaded_filtered_dataset[:5])
