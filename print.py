import unicodedata
import os
import json
from datasets import load_dataset

directory = "/sky-notebook/eval-results/"

scores = {}

skip_model = ["open-aditi-hi-v2-dpo-awq", "Qwen",
              "open-aditi-chat-hi-1.8-awq", "OpenHathi-7B-Hi-v0.1-Base"]

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)

            if file == "metrics.json":
                splits = file_path.replace(directory, "").split('/')
                task = splits[0]
                model = splits[1]
                lang = splits[2]
                file = splits[3]

                if model in skip_model or "awq" in model or "AWQ" in model:
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


def generate_markdown_table(data):
    markdown_output = ""

    # Iterate over tasks and sub-tasks
    task_model_score = {}
    langs = ["hi", "en"]
    for lang in langs:
        lang_dict = data[lang]
        markdown_output += f"#### Language {lang.capitalize()}\n\n"

        tasks = []
        task_model_score[lang] = {}
        for task, tasks_dict in lang_dict.items():
            tasks.append(task)
            if task not in task_model_score[lang]:
                task_model_score[lang][task] = {}
            for model, model_dict in tasks_dict.items():
                if model not in task_model_score[lang][task]:
                    task_model_score[lang][task][model] = {}
                for metric, metric_value in model_dict.items():
                    task_model_score[lang][task][model] = metric_value
                    break

        # Create a table header

        model_scores = {}
        for task, task_dict in task_model_score[lang].items():
            for model, metric_value in task_dict.items():
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(metric_value)

        avg_model_score = {}
        for model, scores in model_scores.items():
            sum = 0
            for s in scores:
                if s > 1:
                    # bluert/chf
                    s = s / 100
                sum += s
            avg = sum / len(scores)
            avg_model_score[model] = avg

        sorted_model_dict = {k: v for k, v in sorted(
            avg_model_score.items(), key=lambda item: item[1], reverse=True)}

        tasks = list(set(tasks))

        taskStr = "| "
        dashStr = "| "
        for task in tasks:
            taskStr += task + " | "
            dashStr += "--- | "

        markdown_output += f"| Model {taskStr} \n"
        markdown_output += f"| --- {dashStr}\n"

        for model, avg in sorted_model_dict.items():
            markdown_output += f"| {model} | "  # {avg:.4f} |
            for task in tasks:
                if model not in task_model_score[lang][task]:
                    markdown_output += f" - |"
                else:
                    average_value = task_model_score[lang][task][model]
                    markdown_output += f" {average_value:.4f} |"

            markdown_output += "\n"

        # Add a newline after the table
        markdown_output += "\n"

    dups = {}
    for lang in langs:
        lang_dict = data[lang]
        for task, tasks_dict in lang_dict.items():
            for model, model_dict in tasks_dict.items():
                for metric, metric_value in model_dict.items():
                    if task not in dups:
                        dups[task] = True
                        if metric == "em_score":
                            metric = "accuracy"
                        markdown_output += f"Task: {task} Metric: {metric} \n\n"

    return markdown_output

# Convert JSON to Markdown table grouped by task and sub-task
markdown_output = generate_markdown_table(sort_data(scores))

# Print the Markdown output
print(markdown_output)

with open("/sky-notebook/eval-results/output.json", 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4)


with open("/sky-notebook/eval-results/output.md", 'w', encoding='utf-8') as file:
    file.write(markdown_output)
