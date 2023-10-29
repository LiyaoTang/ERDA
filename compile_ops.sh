#!/bin/bash
green_start="\033[32m"
green_end="\033[0m"
set -e # exit as soon as any error occur

function info() {
    echo -e "$green_start$1$green_end"
}

# grid_subsampling - single cloud
info "\ncompiling cpp_wrappers\n"
cd ops/cpp_wrappers
bash compile_wrappers.sh

# radius search & grid subsampling - stacked clouds
info "\ncompiling tf_custom_ops\n"
cd ../tf_custom_ops
bash compile_op.sh

# knn search - batched clouds
info "\ncompiling nearest_neighbors\n"
cd ../nearest_neighbors
bash compile_op.sh

# knn search (gpu) - batched clouds
info "\ncompiling knnquery\n"
cd ../knnquery
bash compile_op.sh

# fps sampling - batched clouds
info "\ncompiling sampling\n"
cd ../sampling
bash compile_op.sh

info "\nfinished\n"
cd ../..
