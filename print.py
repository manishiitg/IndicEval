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

print(json.dumps(scores, indent=4))