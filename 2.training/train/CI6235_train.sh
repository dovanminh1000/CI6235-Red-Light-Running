#!/bin/sh
# Usage: 
#CI6235 ./train.sh 1 (to train)
#CI6235 ./trains.sh 0 (to test)

#Specify these!
MODEL_DIR=../../../models/CI6235_RLR_Final_Early_Stopping_6000 #CI6235: Folder contains .config file and output file from training
MODEL_NAME=pipeline

PIPELINE_CONFIG_PATH=${MODEL_DIR}/${MODEL_NAME}.config #CI6235: Path to .config file
NUM_TRAIN_STEPS=200000

SAMPLE_1_OF_N_EVAL_EXAMPLES=1
cd ../tf_object_detection_api/models/research/
echo $PWD

if [ $1 -eq 1 ]
then
    #CI6235: Start training 
    echo "Starting Training..."
    python object_detection/CI6235_model_main.py \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        --model_dir=${MODEL_DIR} \
        --num_train_steps=${NUM_TRAIN_STEPS} \
        --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
        --alsologtostderr
    echo "Finish Training..."

else 
    #CI6235: Start testing
    #CI6235: In pipeline.config file, change RLR_Mixed_val.record to RLR_Mixed_test.record
    echo "Starting Testing..."
    CHECKPOINT_DIR=../../../models/CI6235_RLR_Final_Early_Stopping_6000
    
    python object_detection/CI6235_model_main.py \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        --model_dir=${MODEL_DIR} \
        --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
        --alsologtostderr \
        --run_once=true \
        --checkpoint_dir=${CHECKPOINT_DIR}/
fi
