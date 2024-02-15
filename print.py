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


# Sorting models based on the first metric for each language
sorted_scores_by_lang = {}
for task, task_dict in scores.items():
    for model, model_dict in task_dict.items():
        for lang, lang_dict in model_dict.items():
            print(lang_dict)
            os.exit(1)
            
            # Sort models by the first metric for each language
            sorted_models = sorted(lang_dict.items(), key=lambda x: x[1][first_metric_key])
            
            # Store the sorted models for each language
            if lang not in sorted_scores_by_lang:
                sorted_scores_by_lang[lang] = {}
            
            for model, metrics in sorted_models:
                if model not in sorted_scores_by_lang[lang]:
                    sorted_scores_by_lang[lang][model] = []
                sorted_scores_by_lang[lang][model].append((task, metrics))

print(sorted_scores_by_lang)