# IndicEval


| Model | indicxparaphrase | hellaswag-indic | boolq-hi | xlsum-hi | implicit_hate | indicsentiment | indic-arc-challenge | indicqa | flores | truthfulqa-hi | indicwikibio | indicheadline | indic-arc-easy | mmlu_hi |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| open-aditi-hi-v2-awq |  0.6818 | 0.2420 | 0.8835 | 0.4347 | 40.6529 | 0.9539 | 0.4096 | 0.2178 | 42.4015 | 0.7555 | 0.4788 | 0.4525 | 0.4423 | 0.3158 |
| open-aditi-hi-v2 |  0.6838 | 0.2404 | 0.8541 | 0.4213 | 11.5021 | 0.9729 | 0.4462 | 0.0795 | 43.6822 | 0.6934 | 0.4846 | 0.4565 | 0.4979 | 0.3253 |
| open-aditi-hi-v1-awq |  0.6658 | 0.1673 | 0.5193 | 0.4026 | 37.0739 | 0.8637 | 0.3456 | 0.1144 | 39.0395 | 0.3144 | 0.3411 | 0.4244 | 0.3784 | 0.1687 |
| OpenHermes-2.5-Mistral-7B |  0.8766 | 0.2485 | 0.5979 | 0.1774 | 0.2068 | 0.9048 | 0.3396 | 0.2721 | 30.3465 | 0.3234 | 0.3332 | 0.1996 | 0.3523 | 0.2769 |
| OpenHermes-2.5-Mistral-7B-AWQ |  0.8536 | 0.2479 | 0.5272 | 0.1894 | 6.0594 | 0.9218 | 0.3157 | 0.3116 | 29.3681 | 0.3428 | 0.3067 | 0.2062 | 0.3291 | 0.2750 |
| open-aditi-hi-v1 |  0.5939 | 0.0848 | 0.3758 | 0.4212 | 8.6105 | 0.8798 | 0.3558 | 0.1306 | 40.2376 | 0.4230 | 0.4104 | 0.4248 | 0.3889 | 0.1398 |
| Airavata |  0.3277 | 0.0254 | 0.0373 | 0.4650 | 0.0663 | 0.8437 | 0.0836 | 0.1008 | 58.0555 | 0.2122 | 0.0637 | 0.4346 | 0.1128 | 0.1336 |

#### Language En

| Model | mmlu | truthfulqa | xlsum | arc-challenge | hellaswag | boolq | arc-easy-exact |  
| --- | --- | --- | --- | --- | --- | --- | --- | 
| OpenHermes-2.5-Mistral-7B |  0.5991 | 0.2081 | 0.4328 | 0.7790 | 0.7999 | 0.4061 | 0.8687 |
| OpenHermes-2.5-Mistral-7B-AWQ |  0.5816 | 0.1897 | 0.4317 | 0.7611 | 0.7826 | 0.4199 | 0.8569 |
| open-aditi-hi-v2-awq |  0.5339 | 0.3905 | 0.4307 | 0.7116 | 0.4355 | 0.4401 | 0.8266 |
| open-aditi-hi-v2 |  0.5544 | 0.2999 | 0.4349 | 0.7235 | 0.4738 | 0.3982 | 0.8388 |
| open-aditi-hi-v1 |  0.2597 | 0.3317 | 0.4288 | 0.6271 | 0.3509 | 0.0434 | 0.7588 |
| open-aditi-hi-v1-awq |  0.3149 | 0.2950 | 0.4296 | 0.6024 | 0.3184 | 0.0798 | 0.7361 |
| Airavata |  0.1165 | 0.3586 | 0.4393 | 0.1630 | 0.0277 | 0.0437 | 0.2534 |

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

