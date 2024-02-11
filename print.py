import os
import json

directory = "/sky-notebook/eval-results/"

scores = {}

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            
            splits = file_path.replace(directory, "").split('/')
            task = splits[0]
            sub_task = splits[1]
            model = splits[2]
            shot = splits[3]
            file = splits[4]
            if file == "metrics.json":
                with open(file_path, 'r') as json_file:
                    try:
                        metric = json.load(json_file)
                        
                        if task not in scores:
                            scores[task] = {}
                        if sub_task not in scores[task]:
                            scores[task][sub_task] = {}
                        if shot not in scores[task][sub_task]:
                            scores[task][sub_task][shot] = {}
                        if model not in scores[task][sub_task][shot]:
                            scores[task][sub_task][shot][model] = {}
                        
                        scores[task][sub_task][shot][model] = metric
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in {file}: {e}")
            else:
                print(file_path)

# Function to sort the data
def sort_data(data):
    # List to hold the sorted data
    sorted_data = []
    
    # Iterate over tasks, sub-tasks, shots, and models
    for task, sub_tasks in data.items():
        for sub_task, shots in sub_tasks.items():
            for shot, models in shots.items():
                for model, metrics in models.items():
                    # Check if the metric is available
                    for metric_name, metric_value in metrics.items():
                        # Add the data to the list
                        sorted_data.append((task, sub_task, shot, model, metric_name, metric_value))
                        break  # Break after the first metric is found
    
    # Sort the list based on the metric
    sorted_data.sort(key=lambda x: x[5])
    
    # Create a new JSON structure with the sorted data
    sorted_json = {}
    for task, sub_task, shot, model, metric_name, metric_value in sorted_data:
        if task not in sorted_json:
            sorted_json[task] = {}
        if sub_task not in sorted_json[task]:
            sorted_json[task][sub_task] = {}
        if shot not in sorted_json[task][sub_task]:
            sorted_json[task][sub_task][shot] = {}
        if model not in sorted_json[task][sub_task][shot]:
            sorted_json[task][sub_task][shot][model] = {}
        sorted_json[task][sub_task][shot][model][metric_name] = metric_value
    
    return sorted_json

# Sort the data
sorted_data = sort_data(scores)
print(json.dumps(sorted_data, indent=4))