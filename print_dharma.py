import unicodedata
import os
import json
from datasets import load_dataset

directory = "/sky-notebook/eval-results/dharma"

scores = {}

skip_model = ["open-aditi-hi-v2-dpo-awq", "Qwen",
              "open-aditi-chat-hi-1.8-awq", "OpenHathi-7B-Hi-v0.1-Base"]

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)

            if file == "subject_metrics.json":
                splits = file_path.replace(directory, "").split('/')
                # task = splits[0]
                model = splits[1]
                lang = splits[2]
                file = splits[3]

                if model in skip_model or "awq" in model or "AWQ" in model:
                    continue

                with open(file_path, 'r') as json_file:
                    try:
                        metric = json.load(json_file)
                        for k,v in metric.items():
                            task = k
                            if task not in scores:
                                scores[task] = {}
                            if model not in scores[task]:
                                scores[task][model] = {}
                            if lang not in scores[task][model]:
                                scores[task][model][lang] = {}

                            scores[task][model][lang] = v
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in {file}: {e}")
            else:
                print("skip: ", file_path)


# Function to sort the data
def sort_data(data):
    # List to hold the sorted data
    sorted_data = {}
    # Sorting models based on the first metric for each language
    for task, task_dict in scores.items():
        for model, model_dict in task_dict.items():
            print("model", model, model_dict)
            for lang, lang_dict in model_dict.items():
                for metric, metric_value in lang_dict.items():
                    if lang not in sorted_data:
                        sorted_data[lang] = []

                    if isinstance(metric_value, dict):
                        print("metric", metric, metric_value)
                        continue
                    sorted_data[lang].append(
                        (task, model, metric, metric_value))
                    # break

    for lang, data in sorted_data.items():
        # Sort the list based on the metric
        data.sort(key=lambda x: x[3], reverse=True)

    ret_data = {}
    for lang, data in sorted_data.items():
        for task, model, metric, metric_value in data:
            if lang not in ret_data:
                ret_data[lang] = {}
            if task not in ret_data[lang]:
                ret_data[lang][task] = {}
            if model not in ret_data[lang][task]:
                ret_data[lang][task][model] = {}

            ret_data[lang][task][model][metric] = metric_value

    return ret_data


data = sort_data(scores)
print(json.dumps(data, indent=4))