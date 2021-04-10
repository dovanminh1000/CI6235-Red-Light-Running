# Run from tensorflow/tflite folder 

# Usage: 
#CI6235: ./convert2tflite.sh <options 1=normal model; 0=quantised model>

MODEL_DIR=../models/CI6235_RLR_EarlyStopping

python ../tf_object_detection_api/models/research/object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=${MODEL_DIR}/pipeline.config \
--trained_checkpoint_prefix=${MODEL_DIR}/model.ckpt \
--output_directory=${MODEL_DIR}/tflite \
--add_postprocessing_op=true

if [ $1 -eq 1 ]
then
python tflite_convert.py \
--output_file ${MODEL_DIR}/tflite/CI6235_RLR_Early_Stopping.tflite \
--graph_def_file ${MODEL_DIR}/tflite/tflite_graph.pb \
--input_arrays normalized_input_image_tensor \
--output_arrays TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
--input_shapes 1,300,300,3 \
--allow_custom_ops
else 
python tflite_convert.py \
--output_file ${MODEL_DIR}/tflite/quantized_RLR.tflite \
--graph_def_file ${MODEL_DIR}/tflite/tflite_graph.pb \
--input_arrays normalized_input_image_tensor \
--output_arrays TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
--input_shapes 1,300,300,3 \
--allow_custom_ops \
--inference_type QUANTIZED_UINT8 \
--mean_values 128 \
--std_dev_values 127
fi 
