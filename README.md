# IndicEval


#### Language Hi

| Model | flores | xlsum-hi | implicit_hate | indicqa | truthfulqa-hi | indicheadline | mmlu_hi | indicwikibio | boolq-hi | indicsentiment | indicxparaphrase |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| open-aditi-hi-v2-awq |  42.4015 | 0.4347 | 40.6529 | 0.2178 | 0.7555 | 0.4525 | 0.3158 | 0.4788 | 0.8835 | 0.9539 | 0.6818 |
| open-aditi-hi-v2 |  43.6822 | 0.4213 | 11.5021 | 0.0795 | 0.6934 | 0.4565 | 0.3253 | 0.4846 | 0.8541 | 0.9729 | 0.6838 |
| open-aditi-hi-v1-awq |  39.0395 | 0.4026 | 37.0739 | 0.1144 | 0.3144 | 0.4244 | 0.1687 | 0.3411 | 0.5193 | 0.8637 | 0.6658 |
| OpenHermes-2.5-Mistral-7B |  30.3465 | 0.1774 | 0.2068 | 0.2721 | 0.3234 | 0.1996 | 0.2769 | 0.3332 | 0.5979 | 0.9048 | 0.8766 |
| OpenHermes-2.5-Mistral-7B-AWQ |  29.3681 | 0.1894 | 6.0594 | 0.3116 | 0.3428 | 0.2062 | 0.2750 | 0.3067 | 0.5272 | 0.9218 | 0.8536 |
| open-aditi-hi-v1 |  40.2376 | 0.4212 | 8.6105 | 0.1306 | 0.4230 | 0.4248 | 0.1398 | 0.4104 | 0.3758 | 0.8798 | 0.5939 |
| Airavata |  58.0555 | 0.4650 | 0.0663 | 0.1008 | 0.2122 | 0.4346 | 0.1336 | 0.0637 | 0.0373 | 0.8437 | 0.3277 |

#### Language En

| Model | hellaswag | xlsum | mmlu | arc-easy-exact | truthfulqa |  
| --- | --- | --- | --- | --- | --- | 
| OpenHermes-2.5-Mistral-7B |  0.7999 | 0.4328 | 0.5991 | 0.8687 | 0.2081 |
| OpenHermes-2.5-Mistral-7B-AWQ |  0.7826 | 0.4317 | 0.5816 | 0.8569 | 0.1897 |
| open-aditi-hi-v2-awq |  0.4355 | 0.4307 | 0.5339 | 0.8266 | 0.3905 |
| open-aditi-hi-v2 |  0.4738 | 0.4349 | 0.5544 | 0.8388 | 0.2999 |
| open-aditi-hi-v1 |  0.3509 | 0.4288 | 0.2597 | 0.7588 | 0.3317 |
| open-aditi-hi-v1-awq |  0.3184 | 0.4296 | 0.3149 | 0.7361 | 0.2950 |
| Airavata |  0.0277 | 0.4393 | 0.1165 | 0.2534 | 0.3586 |

Task: flores Metric: chrf 
Task: implicit_hate Metric: chrf 
Task: indicsentiment Metric: accuracy 
Task: boolq-hi Metric: accuracy 
Task: indicxparaphrase Metric: accuracy 
Task: truthfulqa-hi Metric: accuracy 
Task: indicwikibio Metric: bleurt 
Task: xlsum-hi Metric: bleurt 
Task: indicheadline Metric: bleurt 
Task: mmlu_hi Metric: average_acc 
Task: indicqa Metric: accuracy 
Task: arc-easy-exact Metric: accuracy 
Task: hellaswag Metric: accuracy 
Task: mmlu Metric: average_acc 
Task: xlsum Metric: bleurt 
Task: truthfulqa Metric: accuracy


=========

LLM Evaluation for indic models

Supports 
- multi gpu support
- faster inference via vllm
- awq support
- llm judge
- eval on spot instance using skypilot with resume support


sky spot launch -n en-hi-spot eval.yaml
