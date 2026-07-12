import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


class TacticalVisionPipeline:
    def __init__(self, model_id='yolov8n.pt'):
        print("Initializing YOLOv8 Network and ByteTrack Engine...")
        self.model = YOLO(model_id)
        self.tracker = sv.ByteTrack()

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

    def process_video(self, source_path, target_path):
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)

        tracking_data = []

        with sv.VideoSink(target_path, video_info=video_info) as sink:
            for frame_idx, frame in enumerate(frame_generator):
                result = self.model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)

                valid_classes = [0, 32]
                detections = detections[np.isin(detections.class_id, valid_classes)]
                detections = self.tracker.update_with_detections(detections)

                for i in range(len(detections)):
                    bbox = detections.xyxy[i]
                    tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else -1
                    class_id = detections.class_id[i]

                    x_center = (bbox[0] + bbox[2]) / 2
                    y_bottom = bbox[3]

                    tracking_data.append({
                        "frame": frame_idx,
                        "track_id": tracker_id,
                        "class_id": class_id,
                        "pixel_x": x_center,
                        "pixel_y": y_bottom
                    })

                labels = [
                    f"#{tracker_id} {'Ball' if class_id == 32 else 'Player'}"
                    for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
                ]

                annotated_frame = frame.copy()
                annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections,
                                                                labels=labels)

                sink.write_frame(frame=annotated_frame)

        print(f"Vision processing complete. Extracted {len(tracking_data)} spatial data points.")
        return tracking_data, target_path


if __name__ == "__main__":
    vision_engine = TacticalVisionPipeline()