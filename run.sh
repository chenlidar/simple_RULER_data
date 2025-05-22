SEQ_LENGTHS=(
    1024
    2048
    4096
    8192
    16384
    32768
    65536
    131072
)
TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)
ROOT_PATH='RULER'
BENCHMARK='synthetic'
MODEL_NAME='M'
TOKENIZER_TYPE='hf'
MODEL_TEMPLATE_TYPE='base'
NUM_SAMPLES=200
TOKENIZER_PATH=''

for TASK in "${TASKS[@]}"; do
    DATA_DIR="${ROOT_PATH}/${BENCHMARK}/${MODEL_NAME}/${TASK}/data"
    mkdir -p ${DATA_DIR}
    
    for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
        python prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES}
    done
done