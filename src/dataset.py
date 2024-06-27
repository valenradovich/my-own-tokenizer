from datasets import load_dataset, Dataset

amount_sample = 500

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
filtered_dataset.save_to_disk(f'data/fineweb_edu_sample_0.5k')
loaded_filtered_dataset = Dataset.load_from_disk(f'data/fineweb_edu_sample_0.5k')
print(loaded_filtered_dataset[:5])
