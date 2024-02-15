# IndicEval


## Language Hi

| Model | xlsum-hi | indicsentiment | indicwikibio | indicxparaphrase | indicqa | flores | truthfulqa-hi | boolq-hi-exact | implicit_hate |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| open-aditi-hi-v2-awq |  0.4347 | 0.9539 | 0.4788 | 0.6818 | 0.2178 | 42.4015 | 0.7555 | 0.8835 | 40.6529 |
| open-aditi-hi-v1-awq |  0.4026 | 0.8637 | 0.3411 | 0.6658 | 0.1144 | 39.0395 | 0.3144 | 0.5193 | 37.0739 |
| OpenHermes-2.5-Mistral-7B-AWQ |  0.1894 | 0.9218 | 0.3067 | 0.8536 | 0.3116 | 29.3681 | 0.3428 | 0.5272 | 6.0594 |

## Language En

| Model | xlsum | truthfulqa | arc-easy-exact | hellaswag |  
| --- | --- | --- | --- | --- | 
| OpenHermes-2.5-Mistral-7B-AWQ |  0.4317 | 0.1897 | 0.8569 | 0.7826 |
| open-aditi-hi-v2-awq |  0.4307 | 0.3905 | 0.8266 | 0.4355 |
| open-aditi-hi-v1-awq |  0.4296 | 0.2950 | 0.7361 | 0.3184 |

Task: flores Metric: chrf 
Task: implicit_hate Metric: chrf 
Task: indicsentiment Metric: accuracy 
Task: boolq-hi-exact Metric: accuracy 
Task: indicxparaphrase Metric: accuracy 
Task: truthfulqa-hi Metric: accuracy 
Task: indicwikibio Metric: bleurt 
Task: xlsum-hi Metric: bleurt 
Task: indicqa Metric: accuracy 
Task: arc-easy-exact Metric: accuracy 
Task: hellaswag Metric: accuracy 
Task: xlsum Metric: bleurt 
Task: truthfulqa Metric: accuracy 



LLM Evaluation for indic models

Supports 
- multi gpu support
- faster inference via vllm
- awq support
- llm judge
- eval on spot instance using skypilot with resume support


sky spot launch -n en-hi-spot eval.yaml
