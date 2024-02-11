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
    sorted_data.sort(key=lambda x: x[5], reverse=True)
    
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

# Function to convert JSON to Markdown table grouped by model
def json_to_markdown_table(data):
    markdown_output = ""
    
    # Create a dictionary to hold models and their data
    models_data = {}
    
    # Iterate over tasks, sub-tasks, shots, and models to collect data
    for task, sub_tasks in data.items():
        for sub_task, shots in sub_tasks.items():
            for shot, models in shots.items():
                for model, metrics in models.items():
                    # Add the data to the models_data dictionary
                    if model not in models_data:
                        models_data[model] = {}
                    if sub_task not in models_data[model]:
                        models_data[model][sub_task] = {}
                    if shot not in models_data[model][sub_task]:
                        models_data[model][sub_task][shot] = {}
                    # Add the metric to the model's data
                    for metric_name, metric_value in metrics.items():
                        if metric_name not in models_data[model][sub_task][shot]:
                            models_data[model][sub_task][shot][metric_name] = []
                        models_data[model][sub_task][shot][metric_name].append(metric_value)
    
    # Iterate over the models and create a table for each one
    for model, sub_tasks in models_data.items():
        # Add a header for the model
        markdown_output += f"## {model}\n\n"
        
        # Iterate over sub-tasks
        for sub_task, shots in sub_tasks.items():
            # Add a header for the sub-task
            markdown_output += f"### {sub_task.capitalize()}\n\n"
            
            # Create a table header
            markdown_output += "| Shot | Metric | Average Value |\n"
            markdown_output += "| --- | --- | --- |\n"
            
            # Iterate over shots and calculate averages
            for shot, metrics in shots.items():
                for metric_name, metric_values in metrics.items():
                    # Calculate the average
                    average_value = sum(metric_values) / len(metric_values)
                    # Add a row for the shot and metric with the average value
                    markdown_output += f"| {shot} | {metric_name} | {average_value:.4f} |\n"
            
            # Add a newline after the table
            markdown_output += "\n"
    
    return markdown_output

# Convert JSON to Markdown table grouped by model
markdown_output = json_to_markdown_table(sorted_data)

# Print the Markdown output
print(markdown_output)