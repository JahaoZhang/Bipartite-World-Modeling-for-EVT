CHUNKS=30
SAVE_PATH_BASE="exp_results/BWM/dt_one_18000"

# Run splits in order: stt -> dt -> at
SPLITS=(at)

# Set to 1 to log single-process peak GPU memory via torch.cuda.max_memory_allocated()

# For accurate single-GPU peak measurement, use one job on one card, e.g.:
# CHUNKS=1 NUM_GPUS=1 PROCESSES_PER_GPU=1 GPU_IDS=(2)

# 3 GPUs, 4 processes per GPU => 12 concurrent jobs per wave
NUM_GPUS=4
PROCESSES_PER_GPU=4
NUM_PARALLEL=$((NUM_GPUS * PROCESSES_PER_GPU))

# Physical GPU IDs (length must equal NUM_GPUS), e.g. (4 5 6) for cards 4/5/6
GPU_IDS=(4 5 6 7)

TRACK_GPU_MEMORY_ARGS=()
if [ "$TRACK_GPU_MEMORY" = "1" ]; then
    TRACK_GPU_MEMORY_ARGS=(--track-gpu-memory)
fi

run_split() {
    local split=$1
    local save_path="${SAVE_PATH_BASE}/${split}"
    local exp_config="habitat-lab/habitat/config/benchmark/nav/track/track_infer_${split}.yaml"

    echo "============================================================"
    echo "Starting split: ${split}"
    echo "Config: ${exp_config}"
    echo "Save path: ${save_path}"
    echo "============================================================"

    mkdir -p "$save_path"

    local IDX=0
    while [ $IDX -lt $CHUNKS ]; do
        for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
            GPU_SLOT=$((IDX % NUM_GPUS))
            GPU_ID=${GPU_IDS[$GPU_SLOT]}
            echo "Launching ${split} job IDX=$IDX on GPU=$GPU_ID (${PROCESSES_PER_GPU} slots per GPU, wave size $NUM_PARALLEL)"
            CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="habitat-lab" python run_bwm.py \
                --split-num $CHUNKS \
                --split-id $IDX \
                --exp-config "$exp_config" \
                --run-type 'eval' \
                --model-name 'bwm' \
                --save-path "$save_path" \ &
            ((IDX++))
        done
        wait
    done

    echo "Finished split: ${split}"
}

for split in "${SPLITS[@]}"; do
    run_split "$split"
done

echo "All splits finished: ${SPLITS[*]}"
