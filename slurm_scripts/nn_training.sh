#!/bin/bash
#SBATCH --job-name=nn_training_small_256   # Job name   
#SBATCH --output=../logs/%x_job_%j.out # Standard output log (%j expands to jobId)
#SBATCH --error=../logs/%x_job_%j.err  # Standard error log (%j expands to jobId)
#SBATCH --nodes=1                        # Request 1 node
#SBATCH --time=24:00:00                  # Maximum execution time (HH:MM:SS), adjust as needed
#SBATCH --partition=fatq                # Request the 'windq', 'workq' or 'rome' partition, or gpuq for cables

# Capture the start time
start_time=$(date +%s)

echo "----------------------------------------------------"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "----------------------------------------------------"

module purge
source ~/.bashrc
conda activate deep_learning

# --- Run the Python script ---
echo "Running Python script..."
python /home/nicit/deep_learning_course_wfco/scripts/training.py
status=$?

# Error handling
if [ $status -ne 0 ]; then
    echo "Python script failed with exit code $status"
    exit $status
else
    echo "Python script completed successfully"
fi

# Total execution time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo "----------------------------------------------------"
echo "Job finished at $(date)"
echo "Total execution time: $((elapsed_time / 3600)) hours $(((elapsed_time % 3600) / 60)) minutes $((elapsed_time % 60)) seconds"
echo "----------------------------------------------------"