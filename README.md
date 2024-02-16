# IndicEval


#### Language Hi

| Model | flores | indicheadline | xlsum-hi | truthfulqa-hi | boolq-hi | indic-arc-easy | implicit_hate | indicwikibio | indic-arc-challenge | indicqa | hellaswag-indic | indicsentiment | indicxparaphrase | mmlu_hi |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| open-aditi-hi-v2-awq |  42.4015 | 0.4525 | 0.4347 | 0.7555 | 0.8835 | - | 40.6529 | 0.4788 | - | 0.2178 | - | 0.9539 | 0.6818 | 0.3158 |
| open-aditi-hi-v2 |  43.6822 | 0.4565 | 0.4213 | 0.6934 | 0.8541 | 0.4979 | 11.5021 | 0.4846 | 0.4462 | 0.0795 | 0.2404 | 0.9729 | 0.6838 | 0.3253 |
| open-aditi-hi-v1-awq |  39.0395 | 0.4244 | 0.4026 | 0.3144 | 0.5193 | - | 37.0739 | 0.3411 | - | 0.1144 | - | 0.8637 | 0.6658 | 0.1687 |
| OpenHermes-2.5-Mistral-7B-AWQ |  29.3681 | 0.2062 | 0.1894 | 0.3428 | 0.5272 | - | 6.0594 | 0.3067 | - | 0.3116 | - | 0.9218 | 0.8536 | 0.2750 |
| open-aditi-hi-v1 |  40.2376 | 0.4248 | 0.4212 | 0.4230 | 0.3758 | 0.3889 | 8.6105 | 0.4104 | 0.3558 | 0.1306 | - | 0.8798 | 0.5939 | 0.1398 |
| OpenHermes-2.5-Mistral-7B |  30.3465 | 0.1996 | 0.1774 | 0.3234 | 0.5979 | 0.3523 | 0.2068 | 0.3332 | 0.3396 | 0.2721 | 0.2485 | 0.9048 | 0.8766 | 0.2769 |
| Airavata |  58.0555 | 0.4346 | 0.4650 | 0.2122 | 0.0373 | 0.1128 | 0.0663 | 0.0637 | 0.0836 | 0.1008 | 0.0254 | 0.8437 | 0.3277 | 0.1336 |

#### Language En

| Model | mmlu | arc-challenge | xlsum | truthfulqa | arc-easy-exact | hellaswag | boolq |  
| --- | --- | --- | --- | --- | --- | --- | --- | 
| OpenHermes-2.5-Mistral-7B |  0.5991 | 0.7790 | 0.4328 | 0.2081 | 0.8687 | 0.7999 | 0.4061 |
| OpenHermes-2.5-Mistral-7B-AWQ |  0.5816 | - | 0.4317 | 0.1897 | 0.8569 | 0.7826 | - |
| open-aditi-hi-v2 |  0.5544 | 0.7235 | 0.4349 | 0.2999 | 0.8388 | 0.4738 | 0.3982 |
| open-aditi-hi-v2-awq |  0.5339 | - | 0.4307 | 0.3905 | 0.8266 | 0.4355 | - |
| open-aditi-hi-v1-awq |  0.3149 | - | 0.4296 | 0.2950 | 0.7361 | 0.3184 | - |
| open-aditi-hi-v1 |  0.2597 | 0.6271 | 0.4288 | 0.3317 | 0.7588 | 0.3509 | 0.0434 |
| Airavata |  0.1165 | 0.1630 | 0.4393 | 0.3586 | 0.2534 | 0.0277 | 0.0437 |

Task: flores Metric: chrf 
Task: implicit_hate Metric: chrf 
Task: indicsentiment Metric: accuracy 
Task: boolq-hi Metric: accuracy 
Task: indicxparaphrase Metric: accuracy 
Task: truthfulqa-hi Metric: accuracy 
Task: indic-arc-easy Metric: accuracy 
Task: indicwikibio Metric: bleurt 
Task: xlsum-hi Metric: bleurt 
Task: indicheadline Metric: bleurt 
Task: indic-arc-challenge Metric: accuracy 
Task: mmlu_hi Metric: average_acc 
Task: indicqa Metric: accuracy 
Task: hellaswag-indic Metric: accuracy 
Task: arc-easy-exact Metric: accuracy 
Task: hellaswag Metric: accuracy 
Task: arc-challenge Metric: accuracy 
Task: mmlu Metric: average_acc 
Task: xlsum Metric: bleurt 
Task: boolq Metric: accuracy 
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
