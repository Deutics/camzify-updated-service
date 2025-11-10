import secrets
import time
from msilib.schema import File
import io

import cv2


class Notification:
    """
    Registers an intrusion notification event in DB.
    Requires:
      - frame: annotated frame
      - object_id: track ID
      - instance_id: pulled from FrameData
      - line_intrusion_notif_direction: optional (left/right)
    """

    def __init__(self,
                 object_id,
                 frame,
                 instance_id,
                 age="",
                 gender="",
                 event_name=None,
                 rule_name=None,
                 color=None,
                 image=None,
                 line_intrusion_notif_direction=None,
                 all_detections=None):
        self._all_detections = all_detections or []
        self._event_time = time.time()
        self._object_id = str(object_id)
        self._frame = frame
        self._instance_id = instance_id
        self._line_intrusion_notif_direction = line_intrusion_notif_direction

    def _process_frame_for_db(self):
        """
        Converts CV2 frame to in-memory JPEG file for DB storage.
        """
        is_success, im_buf_arr = cv2.imencode(".jpg", self._frame)
        if is_success:
            filename = f"{secrets.token_hex(6)}_{int(self._event_time.timestamp())}.jpg"
            return filename, File(io.BytesIO(im_buf_arr))
        return None

    def _retrieve_db_data(self):
        """
        Return the fields needed to create the InstanceEvent record.
        """
        return {
            "event_time": self._event_time,
            "instance": instance_models.Instance.objects.get(id=self._instance_id),
            "object_id_on_frame": self._object_id,
            "line_intrusion_notif_direction": self._line_intrusion_notif_direction
        }

    def register(self):
        """
        Save notification to DB and trigger background task.
        """
        instance_event = instance_models.InstanceEvent.objects.create(**self._retrieve_db_data())
        processed_frame = self._process_frame_for_db()
        if processed_frame:
            instance_event.frame.save(*processed_frame)

        # Trigger async processing if needed
        background_tasks.register_instance_event_on_nettbox.delay(instance_event.id)

        print(f"[Notification] Registered event for object {self._object_id} on instance {self._instance_id}")
