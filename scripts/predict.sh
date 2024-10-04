# This is suitable for all files which have names starting with predict
python $PREDICT_FILE \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --greedy \
    --output_dir $OUTPUT_DIR \
    --max_new_tokens 20 \
    --is_scored