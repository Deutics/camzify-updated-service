import os
import cv2
from datetime import datetime
from typing import List, Dict, Any
from src.notifications.notification import Notification

class AlertSystem:
    def __init__(self, snapshot_dir: str = "snapshots"):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def handle_intrusion(self, detection_result, intrusion_events: List[Dict[str, Any]], line_points):
        if not intrusion_events:
            return
        ts = datetime.fromtimestamp(detection_result.frame_data.timestamp)
        name = f"intrusion_{detection_result.frame_data.stream_id}_{ts.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        path = os.path.join(self.snapshot_dir, name)
        try:
            annotated = self._annotate(detection_result, intrusion_events, line_points)
            cv2.imwrite(path, annotated)
            print(f"✅ Snapshot saved: {path}")
        except Exception as e:
            print(f"❌ Failed to save snapshot: {e}")

        for ev in intrusion_events:
            print(f"🚨 ALERT: {ev.get('class_name')} on {detection_result.frame_data.stream_id} at {ts.strftime('%H:%M:%S')}")

        # self.register_notification(detection_result, intrusion_events, line_points)

    def _annotate(self, detection_result, intrusion_events, line_points):
        frame = detection_result.frame_data.frame.copy()
        if len(line_points) >= 2:
            cv2.line(frame, tuple(line_points[0]), tuple(line_points[1]), (0,255,255), 3)
        for d in detection_result.detections:
            x1,y1,x2,y2 = map(int, d['bbox'])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            label = f"{d['class_name']}#{d.get('track_id',-1)} {d['confidence']:.2f}"
            cv2.putText(frame,label,(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        for ev in intrusion_events:
            bx = list(map(int, ev['bbox']))
            cv2.rectangle(frame,(bx[0],bx[1]),(bx[2],bx[3]),(0,0,255),4)
            center = (int(ev['position'][0]), int(ev['position'][1]))
            cv2.putText(frame,'INTRUSION',(center[0]-30,center[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)
        ts = datetime.fromtimestamp(detection_result.frame_data.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, ts, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        return frame

    def register_notification(self, detection_result, intrusion_events, line_points):
        if not intrusion_events:
            return

        annotated = self._annotate(detection_result, intrusion_events, line_points)

        # Build a lightweight list of all objects in the frame
        all_detections = [
            {
                "bbox": d["bbox"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "track_id": d.get("track_id")
            }
            for d in detection_result.detections
        ]

        for ev in intrusion_events:
            Notification(
                object_id=ev["track_id"],
                frame=annotated,
                instance_id=ev["instance_id"],
                event_name=ev.get("rule_type"),
                rule_name=ev.get("rule_type"),
                line_intrusion_notif_direction=ev.get("direction"),
                all_detections=all_detections  # NEW
            ).register()