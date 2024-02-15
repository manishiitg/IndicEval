# IndicEval


## Language En

| Model | hellaswag | xlsum | arc-easy-exact | truthfulqa | 
| --- | --- | --- | --- | --- | 
| open-aditi-hi-v2-awq | 0.4355 | 0.4307 | 0.8266 | 0.3905 |
| open-aditi-hi-v1-awq | 0.3184 | 0.4296 | 0.7361 | 0.2950 |
| OpenHermes-2.5-Mistral-7B-AWQ | 0.7826 | 0.4317 | 0.8569 | 0.1897 |

## Language Hi

| Model | indicxparaphrase | boolq-hi-exact | indicqa | truthfulqa-hi | xlsum-hi | indicsentiment | flores | indicwikibio | implicit_hate | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| open-aditi-hi-v2-awq | 0.6818 | 0.8835 | 0.2178 | 0.7555 | 0.4347 | 0.9539 | 42.4015 | 0.4788 | 40.6529 |
| open-aditi-hi-v1-awq | 0.6658 | 0.5193 | 0.1144 | 0.3144 | 0.4026 | 0.8637 | 39.0395 | 0.3411 | 37.0739 |
| OpenHermes-2.5-Mistral-7B-AWQ | 0.8536 | 0.5272 | 0.3116 | 0.3428 | 0.1894 | 0.9218 | 29.3681 | 0.3067 | 6.0594 |



LLM Evaluation for indic models

Supports 
- multi gpu support
- faster inference via vllm
- awq support
- llm judge
- eval on spot instance using skypilot with resume support


sky spot launch -n en-hi-spot eval.yaml
