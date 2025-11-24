#!/bin/bash
#SBATCH --partition=exercise-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

spack env activate cuda
spack load cuda@12.4.0

OUTPUT_FILE="5_2.csv"

# Header: s = problem size, t = threads-per-block, then the program output fields,
# plus total_ms = memcpy_h2d_ms + memcpy_d2h_ms + t_matmul_ms
echo "s,t,grid_dim,block_dim,memcpy_h2d_ms,bandwith_h2d_gbps,memcpy_d2h_ms,bandwidth_d2h_gbps,t_matmul_ms,total_ms,speedup" > $OUTPUT_FILE

# Threads-per-block (t x t <= 1k)
T_LIST=(1 2 4 8 10 16 25 32)

# Problem sizes
S_LIST=(128 256 512 800 1024 2048 4096 5000)

for s in "${S_LIST[@]}"; do
  # run host baseline once per problem size
  echo "Running host baseline: s=${s}"
  HOST_OUT=$(./bin/matMul --host -s ${s} 2>/dev/null)
  HOST_MS=$(echo "$HOST_OUT" | awk '{print $1}')
  if [ -z "$HOST_MS" ]; then
    echo "ERROR: no host output for s=${s}" >&2
    HOST_MS="NA"
  fi
  for t in "${T_LIST[@]}"; do
    echo "Running: s=${s}, t=${t}"
    # run the program and capture its single-line CSV output
    LINE=$(./bin/matMul --no-check -s ${s} -t ${t})
    if [ -z "$LINE" ]; then
      echo "ERROR: no GPU output for s=${s},t=${t}" >&2
      continue
    fi

    # parse relevant numeric fields from program CSV: field3=h2d_ms, field5=d2h_ms, field7=mat_ms
    H2D_MS=$(echo "$LINE" | awk -F',' '{print $3}')
    D2H_MS=$(echo "$LINE" | awk -F',' '{print $5}')
    MAT_MS=$(echo "$LINE" | awk -F',' '{print $7}')

    # compute total (h2d + d2h + mat)
    TOTAL_MS=$(awk -v a="$H2D_MS" -v b="$D2H_MS" -v c="$MAT_MS" 'BEGIN{printf "%.6f", (a+0)+(b+0)+(c+0)}')

    # compute speedups; guard against zero or NA
    if [ "$MAT_MS" = "0" ] || [ -z "$MAT_MS" ] || [ "$MAT_MS" = "NA" ] || [ "$HOST_MS" = "NA" ]; then
      SPEED_NO="NA"
    else
      SPEED_NO=$(awk -v h="$HOST_MS" -v m="$MAT_MS" 'BEGIN{ if(m>0) printf "%.6f", h/m; else print "NA" }')
    fi

    if [ "$TOTAL_MS" = "0" ] || [ -z "$TOTAL_MS" ] || [ "$TOTAL_MS" = "NA" ] || [ "$HOST_MS" = "NA" ]; then
      SPEED_WITH="NA"
    else
      SPEED_WITH=$(awk -v h="$HOST_MS" -v t="$TOTAL_MS" 'BEGIN{ if(t>0) printf "%.6f", h/t; else print "NA" }')
    fi

    # append to CSV: s,t,<program CSV fields>,total_ms,host_ms,speedups
    echo "${s},${t},${LINE},${TOTAL_MS},${HOST_MS},${SPEED_NO},${SPEED_WITH}" >> "${OUTPUT_FILE}"
  done
done