# bash scripts/indic_eval/warmup.sh &
bash scripts/prepare_eval_data.sh
bash scripts/indic_eval/indicxparaphrase.sh
bash scripts/indic_eval/truthfulqa.sh
bash scripts/indic_eval/xlsum.sh
bash scripts/indic_eval/boolq.sh
bash scripts/indic_eval/arc.sh
bash scripts/indic_eval/flores.sh
bash scripts/indic_eval/hellaswag.sh
bash scripts/indic_eval/in22_gen.sh

bash scripts/indic_eval/indicheadline.sh
bash scripts/indic_eval/indicqa.sh
bash scripts/indic_eval/indicsentiment.sh
bash scripts/indic_eval/indicwikibio.sh
bash scripts/indic_eval/mmlu.sh
python3 print.py