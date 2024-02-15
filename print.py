import os
import json

directory = "/sky-notebook/eval-results/"

scores = {}

skip_model = ["open-aditi-hi-v2-dpo-awq"]

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)

            print(file_path)
            if file == "metrics.json":
                splits = file_path.replace(directory, "").split('/')
                task = splits[0]
                model = splits[1]
                lang = splits[2]
                file = splits[3]

                if model in skip_model:
                    continue

                with open(file_path, 'r') as json_file:
                    try:
                        metric = json.load(json_file)

                        if task not in scores:
                            scores[task] = {}
                        if model not in scores[task]:
                            scores[task][model] = {}
                        if lang not in scores[task][model]:
                            scores[task][model][lang] = {}
                        
                        scores[task][model][lang] = metric
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in {file}: {e}")
            else:
                print(file_path)


# Function to sort the data
def sort_data(data):
    # List to hold the sorted data
    sorted_data = {}
    # Sorting models based on the first metric for each language
    for task, task_dict in scores.items():
        for model, model_dict in task_dict.items():
            for lang, lang_dict in model_dict.items():
                for metric, metric_value in lang_dict.items():
                    if lang not in sorted_data:
                        sorted_data[lang] = []
                    sorted_data[lang].append((task, model, metric, metric_value))
                    break  # Break after the first metric is found
                
        
    for lang, data in sorted_data.items():
        # Sort the list based on the metric
        sorted_data[lang] = data.sort(key=lambda x: x[3], reverse=True)            

    return sorted_data

print(sort_data(scores))