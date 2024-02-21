# IndicEval


#### Language Hi

| Model | implicit_hate | truthfulqa-hi | hellaswag-indic | indicxparaphrase | boolq-hi | flores | indicheadline | indic-arc-challenge | mmlu_hi | indicsentiment | xlsum-hi | indic-arc-easy | indicqa | indicwikibio |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| open-aditi-hi-v2-awq |  40.6529 | 0.7555 | 0.2420 | 0.6818 | 0.8835 | 42.4015 | 0.4525 | 0.4096 | 0.3158 | 0.9539 | 0.4347 | 0.4423 | 0.2178 | 0.4788 |
| open-aditi-hi-v2 |  11.5021 | 0.6934 | 0.2404 | 0.6838 | 0.8541 | 43.6822 | 0.4565 | 0.4462 | 0.3253 | 0.9729 | 0.4213 | 0.4979 | 0.0795 | 0.4846 |
| open-aditi-hi-v3 |  8.8315 | 0.5369 | 0.4891 | 0.8846 | 0.5401 | 48.2859 | 0.4682 | 0.4633 | 0.1351 | 0.9519 | 0.4490 | 0.5480 | 0.0058 | 0.5034 |
| open-aditi-hi-v1-awq |  37.0739 | 0.3144 | 0.1673 | 0.6658 | 0.5193 | 39.0395 | 0.4244 | 0.3456 | 0.1687 | 0.8637 | 0.4026 | 0.3784 | 0.1144 | 0.3411 |
| OpenHermes-2.5-Mistral-7B |  0.2068 | 0.3234 | 0.2485 | 0.8766 | 0.5979 | 30.3465 | 0.1996 | 0.3396 | 0.2769 | 0.9048 | 0.1774 | 0.3523 | 0.2721 | 0.3332 |
| OpenHermes-2.5-Mistral-7B-AWQ |  6.0594 | 0.3428 | 0.2479 | 0.8536 | 0.5272 | 29.3681 | 0.2062 | 0.3157 | 0.2750 | 0.9218 | 0.1894 | 0.3291 | 0.3116 | 0.3067 |
| open-aditi-hi-v1 |  8.6105 | 0.4230 | 0.0848 | 0.5939 | 0.3758 | 40.2376 | 0.4248 | 0.3558 | 0.1398 | 0.8798 | 0.4212 | 0.3889 | 0.1306 | 0.4104 |
| Airavata |  6.3612 | 0.0466 | 0.0550 | 0.6419 | 0.0128 | 58.5260 | 0.4346 | 0.0836 | 0.1336 | 0.0992 | 0.4650 | 0.1128 | 0.0155 | 0.0637 |

#### Language En

| Model | hellaswag | arc-challenge | mmlu | xlsum | arc-easy-exact | boolq | truthfulqa |  
| --- | --- | --- | --- | --- | --- | --- | --- | 
| OpenHermes-2.5-Mistral-7B |  0.7999 | 0.7790 | 0.5991 | 0.4328 | 0.8687 | 0.4061 | 0.2081 |
| OpenHermes-2.5-Mistral-7B-AWQ |  0.7826 | 0.7611 | 0.5816 | 0.4317 | 0.8569 | 0.4199 | 0.1897 |
| open-aditi-hi-v3 |  0.7645 | 0.7415 | 0.5478 | 0.4352 | 0.8384 | 0.3749 | 0.3097 |
| open-aditi-hi-v2-awq |  0.4355 | 0.7116 | 0.5339 | 0.4307 | 0.8266 | 0.4401 | 0.3905 |
| open-aditi-hi-v2 |  0.4738 | 0.7235 | 0.5544 | 0.4349 | 0.8388 | 0.3982 | 0.2999 |
| open-aditi-hi-v1 |  0.3509 | 0.6271 | 0.2597 | 0.4288 | 0.7588 | 0.0434 | 0.3317 |
| open-aditi-hi-v1-awq |  0.3184 | 0.6024 | 0.3149 | 0.4296 | 0.7361 | 0.0798 | 0.2950 |
| Airavata |  0.1799 | 0.1630 | 0.1165 | 0.4393 | 0.6772 | 0.5086 | 0.3574 |

Task: flores Metric: chrf 

Task: implicit_hate Metric: chrf 

Task: indicsentiment Metric: accuracy 

Task: indicxparaphrase Metric: accuracy 

Task: boolq-hi Metric: accuracy 

Task: truthfulqa-hi Metric: accuracy 

Task: indic-arc-easy Metric: accuracy 

Task: indicwikibio Metric: bleurt 

Task: hellaswag-indic Metric: accuracy 

Task: indicheadline Metric: bleurt 

Task: xlsum-hi Metric: bleurt 

Task: indic-arc-challenge Metric: accuracy 

Task: mmlu_hi Metric: average_acc 

Task: indicqa Metric: accuracy 

Task: arc-easy-exact Metric: accuracy 

Task: hellaswag Metric: accuracy 

Task: arc-challenge Metric: accuracy 

Task: mmlu Metric: average_acc 

Task: boolq Metric: accuracy 

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


===========

To run this via skypilot https://github.com/skypilot-org/skypilot use


`sky spot launch -n en-hi-spot eval.yaml`

=========== 
To run this on machine having GPU look at eval.yaml



add your model name in scripts/indic_eval/commaon_vars.sh to evalulate and run scripts/indic_eval/run_suite.sh

