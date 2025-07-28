#!/usr/bin/env bash

# Exit on undefined variable, but NOT on command failure
set -u

# Activate conda environment
conda activate dsg

# Define parameter arrays
recordings=(cooking kitchen2 office1 office2 office3)
functions=(farthest_point sampling_cc)
masks=(true false)

# Base output directory
base_out="./outputs/experiments"

# Ensure the base output directory exists
mkdir -p "$base_out"

# ########################################
# # 1) Undistort each recording
# ########################################
echo "===== STARTING UNDISTORTION ====="
for rec in "${recordings[@]}"; do
  echo "-> Undistorting recording: recording=${rec}"
  python src/undistort.py "$rec"
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "⚠️  undistort.py failed for ${rec} (exit code $exit_code)" >&2
  else
    echo "✅  undistort.py succeeded for ${rec}"
  fi
  echo
done

########################################
# 2) Run all experiment combinations
########################################
echo "===== STARTING EXPERIMENTS ====="
for rec in "${recordings[@]}"; do
  for func in "${functions[@]}"; do
    for mask in "${masks[@]}"; do

      # Build output name and relative dir
      output_name="${rec}_${func}_mask${mask}"
      rel_dir="${base_out}/${output_name}"
      mkdir -p "$rel_dir"

      # Convert to absolute path
      abs_dir="$(realpath "$rel_dir")"

      # Report what we're running
      echo "-> Experiment: recording=${rec}, fct=${func}, masks=${mask}"
      echo "   Output folder (absolute): ${abs_dir}"

      python src/sam2_reinit.py \
        recording="${rec}" \
        output_folder="$abs_dir" \
        new_objects_fct="${func}" \
        prompt_with_masks="${mask}" \
        subsample=5 \
        chunk_size=10 \
        overlap=1
      exit_code=$?

      # Report status
      if [ $exit_code -ne 0 ]; then
        echo "⚠️  Experiment failed: ${output_name} (exit code $exit_code)" >&2
      else
        echo "✅  Experiment succeeded: ${output_name}"
      fi
      echo

    done
  done
done

echo "===== ALL DONE ====="
