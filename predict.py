from ultralytics import YOLO
import cv2
import h5py
import numpy as np

# Load a model
model = YOLO('/localhome/zzhang/sources/yolo_events/yolov8/runs/detect/train21/weights/best.pt')  # load a partially trained model

# events_file = h5py.File("/mnt/celeste/projects/machine_learning/datasets/optex/full_sequence/test/approach_angle_13.h5", "r+")
events_file = h5py.File("/mnt/ssd1/tmp/zzhang/test/dt_30000/recording_2023-09-22_13-28-55.h5", "r+")

num_ev_histograms = len(events_file['data'])

out = cv2.VideoWriter('/mnt/ssd1/tmp/zzhang/test/output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (1280, 720))
for idx in range(num_ev_histograms):
    ev_histo = np.transpose(events_file['data'][idx], (1, 2, 0))
    results = model(ev_histo, iou=0.65)  # return a generator of Results objects
    annotated_frame = results[0].plot()
    out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
out.release()