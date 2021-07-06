python3 main.py -h
usage: main.py [-h] --model-path MODEL_PATH --label-path LABEL_PATH
               --source-video-path SOURCE_VIDEO_PATH
               [--output-video-path OUTPUT_VIDEO_PATH]
               [--sampling-rate SAMPLING_RATE] [--confidence CONFIDENCE]

Start the model for counting vehicles on the road.

optional arguments:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH, -mp MODEL_PATH
                        Path to the model.
  --label-path LABEL_PATH, -lp LABEL_PATH
                        Path to the labels. ./*.pbtxt
  --source-video-path SOURCE_VIDEO_PATH, -sp SOURCE_VIDEO_PATH
                        Path to the video source.
  --output-video-path OUTPUT_VIDEO_PATH, -op OUTPUT_VIDEO_PATH
                        Path to the video output with all the labels.
  --sampling-rate SAMPLING_RATE, -sr SAMPLING_RATE
                        Frame processing rate, could be set lower on faster
                        GPUs. Default value: 10
  --confidence CONFIDENCE, -c CONFIDENCE
                        Confidence of the objects that will be counted.
                        Default value: 0.65

Example: python3 main.py -mp ./exported_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8_retrained/saved_model/ -lp annotations/label_map.pbtxt -sp src/video/1.mp4 -op ./
