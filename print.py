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
                sub_task = splits[1]
                model = splits[2]
                shot = splits[3]
                file = splits[4]

                if model in skip_model:
                    continue
                
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
    # Dictionary to hold the data grouped by model
    grouped_data = {}
    
    # Iterate over tasks, sub-tasks, shots, and models
    for task, sub_tasks in data.items():
        for sub_task, shots in sub_tasks.items():
            for shot, models in shots.items():
                for model, metrics in models.items():
                    # Check if the metric is available
                    for metric_name, metric_value in metrics.items():
                        # Add the data to the dictionary
                        if model not in grouped_data:
                            grouped_data[model] = []
                        grouped_data[model].append((task, sub_task, shot, model, metric_name, metric_value))
                        break
    
    # Sort the data for each model based on the metric
    for model, data_list in grouped_data.items():
        data_list.sort(key=lambda x: x[6], reverse=True)
    
    # Create a new JSON structure with the sorted data
    sorted_json = {}
    for model, data_list in grouped_data.items():
        for task, sub_task, shot, model, metric_name, metric_value in data_list:
            if model not in sorted_json:
                sorted_json[model] = {}
            if task not in sorted_json[model]:
                sorted_json[model][task] = {}
            if sub_task not in sorted_json[model][task]:
                sorted_json[model][task][sub_task] = {}
            if shot not in sorted_json[model][task][sub_task]:
                sorted_json[model][task][sub_task][shot] = {}
            sorted_json[model][task][sub_task][shot][metric_name] = metric_value
    
    return sorted_json

# Sort the data
sorted_data = sort_data(scores)
print(json.dumps(sorted_data, indent=4))

# Function to convert JSON to Markdown table grouped by task and sub-task
def json_to_markdown_table(data):
    markdown_output = ""
    
    # Iterate over tasks and sub-tasks
    for task, sub_tasks in data.items():
        for sub_task, shots in sub_tasks.items():
            # Add a header for the task and sub-task
            markdown_output += f"## {task.capitalize()} - {sub_task.capitalize()}\n\n"
            
            # Create a table header
            markdown_output += "| Model | Metric | Average Value |\n"
            markdown_output += "| --- | --- | --- |\n"
            
            # Collect metrics for all shots
            all_metrics = {}
            for shot, models in shots.items():
                for model, metrics in models.items():
                    for metric_name, metric_value in metrics.items():
                        if model not in all_metrics:
                            all_metrics[model] = {}
                        if metric_name not in all_metrics[model]:
                            all_metrics[model][metric_name] = []
                        all_metrics[model][metric_name].append(metric_value)
            
            # Calculate the average for each metric across all shots
            for model, metrics in all_metrics.items():
                for metric_name, metric_values in metrics.items():
                    average_value = sum(metric_values) / len(metric_values)
                    markdown_output += f"| {model} | {metric_name} | {average_value:.4f} |\n"
            
            # Add a newline after the table
            markdown_output += "\n"
    
    return markdown_output

# Convert JSON to Markdown table grouped by task and sub-task
markdown_output = json_to_markdown_table(sorted_data)

# Print the Markdown output
print(markdown_output)

# Function to save Markdown output to a file
def save_markdown_to_file(markdown_text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(markdown_text)

# Function to save sorted JSON data to a file
def save_json_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# Save the Markdown output to a file
save_markdown_to_file(markdown_output, directory + 'output.md')
# Save the sorted JSON data to a file
save_json_to_file(sorted_data, directory + 'sorted_data.json')


def sort_data(data):
    # List to hold the sorted data
    sorted_data = []
    
    # Iterate over tasks, sub-tasks, shots, and models
    for task, sub_tasks in data.items():
        for sub_task, shots in sub_tasks.items():
            for shot, models in shots.items():
                for model, metrics in models.items():
                    # Check if the metric is available
                    model_metrics = []
                    for metric_name, metric_value in metrics.items():
                        # Add the metric to the list
                        model_metrics.append((metric_name, metric_value))
                    
                    # Sort the metrics by name
                    model_metrics.sort()
                    
                    # Add the data to the list
                    sorted_data.append((task, sub_task, shot, model, model_metrics))
    
    # Sort the list based on the metrics
    sorted_data.sort(key=lambda x: [(metric_name, metric_value) for metric_name, metric_value in x[4]], reverse=True)
    
    # Create a new JSON structure with the sorted data
    sorted_json = {}
    for task, sub_task, shot, model, model_metrics in sorted_data:
        if task not in sorted_json:
            sorted_json[task] = {}
        if sub_task not in sorted_json[task]:
            sorted_json[task][sub_task] = {}
        if shot not in sorted_json[task][sub_task]:
            sorted_json[task][sub_task][shot] = {}
        if model not in sorted_json[task][sub_task][shot]:
            sorted_json[task][sub_task][shot][model] = {}
        for metric_name, metric_value in model_metrics:
            sorted_json[task][sub_task][shot][model][metric_name] = metric_value
    
    return sorted_json

# Sort the data
sorted_data = sort_data(scores)
print(json.dumps(sorted_data, indent=4))

def json_to_markdown_table(sorted_data):
    markdown_output = ""
    
    # Iterate over tasks and sub-tasks
    for task, sub_tasks in sorted_data.items():
        for sub_task, shots in sub_tasks.items():
            # Add a header for the task and sub-task
            markdown_output += f"## {task.capitalize()} - {sub_task.capitalize()}\n\n"
            
            # Create a table header
            markdown_output += "| Model | Metric | Average Value |\n"
            markdown_output += "| --- | --- | --- |\n"
            
            # Collect metrics for all shots and models
            all_metrics = {}
            for shot, models in shots.items():
                for model, metrics in models.items():
                    for metric_name, metric_value in metrics.items():
                        if model not in all_metrics:
                            all_metrics[model] = {}
                        if metric_name not in all_metrics[model]:
                            all_metrics[model][metric_name] = []
                        all_metrics[model][metric_name].append(metric_value)
            
            # Calculate the average for each metric across all shots and models
            for model, metrics in all_metrics.items():
                for metric_name, metric_values in metrics.items():
                    # Ensure metric_values is a list of numbers before calculating the average
                    if isinstance(metric_values, list) and all(isinstance(x, (int, float)) for x in metric_values):
                        average_value = sum(metric_values) / len(metric_values)
                        markdown_output += f"| {model} | {metric_name} | {average_value:.4f} |\n"
                    else:
                        markdown_output += f"| {model} | {metric_name} | N/A (non-numeric data) |\n"
            
            # Add a newline after the table
            markdown_output += "\n"
    
    return markdown_output

# Convert JSON to Markdown table grouped by task and sub-task
markdown_output = json_to_markdown_table(sorted_data)

# Print the Markdown output
print(markdown_output)

# Function to save Markdown output to a file
def save_markdown_to_file(markdown_text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(markdown_text)

# Function to save sorted JSON data to a file
def save_json_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# Save the Markdown output to a file
save_markdown_to_file(markdown_output, directory + 'output-all.md')
# Save the sorted JSON data to a file
save_json_to_file(sorted_data, directory + 'sorted_data-all.json')