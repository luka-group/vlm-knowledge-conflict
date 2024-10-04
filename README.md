# VLM Knowledge Conflict

![Static Badge](https://img.shields.io/badge/vision%20and%20language-blue)
![Static Badge](https://img.shields.io/badge/knowledge%20conflict-blue)

Code and data for paper [Unravelling Cross-Modality Knowledge Conflicts in Large Vision-Language Models]().

![](figures\case.jpg)

<p align="center">
    [<a href="https://darthzhu.github.io/cross-modality-knowledge-conflict/">Website</a>] •
    [<a href="">Paper</a>] •
    [<a href="https://huggingface.co/datasets/DarthZhu/vlm-knowledge-conflict">Dataset</a>] •
    [<a href="">Twitter</a>]
</p>

## Data Preview

```json
{
    "id": "infoseek_val_00068862",
    "entity": "Arctic hare",
    "question": "What is the closest upper taxonomy of this animal?",
    "image": "oven_05050169.jpg",
    "multiple_choices": {
        "A": "Lepus",
        "B": "Marmota",
        "C": "Oryctolagus",
        "D": "Sylvilagus"
    },
    "multiple_choices_answer": "A"
}
```

## Analysis

To conduct the analyses in our paper, run codes in `src/analysis`.
We provide codes for answer prediction, and contrastive metric.
You can refer to the scripts in `scripts/`.

For answer prediction, run the following command:

```bash
python $PREDICT_FILE \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --greedy \
    --max_new_tokens 20 \
    --is_scored
```

To analyze a new model: if the model is supported by vllm, use `src/analysis/predict_vllm.py` for faster prediction; otherwise, implement it in `src/models/local.py` and run `src/analysis/predict.py`.

For contrastive metric, you should first generate both textual and visual logits, then run the following command:

```bash
python src/analysis/post_hoc_contrastive_decoding_metric.py \
    --dataset $DATASET \
    --model_name $MODEL_NAME
```

## Improvements

To run dynamic contrastive decoding, use the following command:

```bash
python src/inference_time/post_hoc_contrastive_decoding.py \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --method dynamic
```

## Evaluation

To calculate the accuracy (Acc), run the following command:

```bash
python src/evaluate/evaluate_mc.py \
    --dataset $DATASET \
    --input_file $FILE_PATH
```

To calculate the recognized accuracy (R. Acc), run the following command:

```bash
python src/evaluate/evaluate_mc.py \
    --dataset $DATASET \
    --input_file $FILE_PATH \
    --cleaned_model $MODEL_NAME
```

To calculate the flip rate (FR), run the following command:

```bash
python src/evaluate/evaluate_mc.py \
    --dataset $DATASET \
    --input_file $FILE_PATH \
    --input_file_2 $FILE_PATH_2 \
    --cleaned_model $MODEL_NAME
```

## Citation

**TBA**

<!-- If you find this repo useful, please cite the following paper:

```bib
@article{,
    title={Unravelling Cross-Modality Knowledge Conflicts in Large Vision-Language Models},
    author={},
    journal={arXiv preprint},
    year={2024}
}
``` -->