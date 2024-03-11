# IndicEval


=========

LLM Evaluation for indic models mainly for hindi language

easy evalulate your models on hindi language own your own GPU's

Supports 
- multi gpu support
- faster inference via vllm
- awq support


Dataset's for evaluation: https://huggingface.co/collections/manishiitg/indiceval-65eed3f98c0e239a9b0a4eae


===========

To run this via skypilot https://github.com/skypilot-org/skypilot use


`sky spot launch -n en-hi-spot eval.yaml`

=========== 

To run this on machine having GPU look at eval.yaml

add your model name in scripts/indic_eval/commaon_vars.sh to evalulate and run scripts/indic_eval/run_suite.sh


| Model | xlsum-hi | truthfulqa-hi | indic-arc-easy | mmlu_hi | indicqa | flores | indicheadline | indicxparaphrase | hellaswag-indic | indicwikibio | boolq-hi | implicit_hate | indic-arc-challenge | indicsentiment |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| open-aditi-hi-v2 |  0.4213 | 0.6934 | 0.4979 | 0.3253 | 0.0795 | 43.6822 | 0.4565 | 0.6838 | 0.2404 | 0.4846 | 0.8541 | 11.5021 | 0.4462 | 0.9729 |
| open-aditi-hi-v3 |  0.4490 | 0.5369 | 0.5480 | 0.1351 | 0.0058 | 48.2859 | 0.4682 | 0.8846 | 0.4891 | 0.5034 | 0.5401 | 8.8315 | 0.4633 | 0.9519 |
| open-aditi-hi-v4 |  0.4046 | 0.7671 | 0.4529 | 0.2124 | 0.0026 | 47.8500 | 0.1980 | 0.7737 | 0.3595 | 0.4894 | 0.7015 | 5.9709 | 0.3857 | 0.9699 |
| OpenHermes-2.5-Mistral-7B |  0.1774 | 0.3234 | 0.3523 | 0.2769 | 0.2721 | 30.3465 | 0.1996 | 0.8766 | 0.2485 | 0.3332 | 0.5979 | 0.2068 | 0.3396 | 0.9048 |
| OpenHermes-2.5-Mistral-7B-AWQ |  0.1894 | 0.3428 | 0.3291 | 0.2750 | 0.3116 | 29.3681 | 0.2062 | 0.8536 | 0.2479 | 0.3067 | 0.5272 | 6.0594 | 0.3157 | 0.9218 |
| open-aditi-hi-v1 |  0.4212 | 0.4230 | 0.3889 | 0.1398 | 0.1306 | 40.2376 | 0.4248 | 0.5939 | 0.0848 | 0.4104 | 0.3758 | 8.6105 | 0.3558 | 0.8798 |
| Airavata |  0.4650 | 0.0466 | 0.1128 | 0.1336 | 0.0155 | 58.5260 | 0.4346 | 0.6419 | 0.0550 | 0.0637 | 0.0128 | 6.3612 | 0.0836 | 0.0992 |

#### Language En

| Model | boolq | truthfulqa | arc-easy-exact | mmlu | hellaswag | xlsum | arc-challenge |  
| --- | --- | --- | --- | --- | --- | --- | --- | 
| open-aditi-hi-v4 |  0.3905 | 0.3378 | 0.8460 | 0.5725 | 0.7603 | 0.4384 | 0.7491 |
| OpenHermes-2.5-Mistral-7B |  0.4061 | 0.2081 | 0.8687 | 0.5991 | 0.7999 | 0.4328 | 0.7790 |
| OpenHermes-2.5-Mistral-7B-AWQ |  0.4199 | 0.1897 | 0.8569 | 0.5816 | 0.7826 | 0.4317 | 0.7611 |
| open-aditi-hi-v3 |  0.3749 | 0.3097 | 0.8384 | 0.5478 | 0.7645 | 0.4352 | 0.7415 |
| open-aditi-hi-v2 |  0.3982 | 0.2999 | 0.8388 | 0.5544 | 0.4738 | 0.4349 | 0.7235 |
| open-aditi-hi-v1 |  0.0434 | 0.3317 | 0.7588 | 0.2597 | 0.3509 | 0.4288 | 0.6271 |
| Airavata |  0.5086 | 0.3574 | 0.6772 | 0.1165 | 0.1799 | 0.4393 | 0.1630 |

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

