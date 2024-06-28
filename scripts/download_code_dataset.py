"""download python code dataset and filter it to 'x' samples"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset, Dataset

amount_sample = 100_000

dataset = load_dataset("ArtifactAI/arxiv_deep_learning_python_research_code_functions_summaries", split='train', streaming=True)

dataset_sample = []

for i, example in enumerate(dataset):
    dataset_sample.append(example['full_code'])
    if i == amount_sample:
        break
    
print('samples filtered:', len(dataset_sample))

# Convert to Dataset
filtered_dataset = Dataset.from_dict({'text': dataset_sample})
print('dataset filtered done:', len(filtered_dataset))  

dataset_name = 'python_code'

# Save and load the filtered dataset locally
filtered_dataset.save_to_disk(f'data/{dataset_name}')
loaded_filtered_dataset = Dataset.load_from_disk(f'data/{dataset_name}')
#print(loaded_filtered_dataset[:5])
