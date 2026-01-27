"""VAF Multistream Processing - Parallel Capture, Single Detection, Feature Classification"""

import time
from stream.multistream_manager import MultiStreamManager
from core.frame_processor import FrameProcessor
from features.line_intrusion_detector import LineIntrusionDetector
from utils_main.logger import get_logger

logger = get_logger(__name__)


def main():

    stream_sources = {
        "74": "rtsp://media.camzify.live:8554/74",
        "73": "rtsp://media.camzify.live:8554/73",
    }
    
    enable_display = True
    model_path = "yolo11n_custom.pt"

    # Stream Manager (Parallel Capture)
    stream_manager = MultiStreamManager(stream_sources, enable_display=enable_display)
    
    # Frame Processor (ONE model for all streams)
    frame_processor = FrameProcessor(
        model_path=model_path,
        conf_thresh=0.45
    )
    
    # Configure features per stream (NO models, just classifiers)
    stream_features = {}
    
    # Camera 1: Horizontal line, both directions
    stream_features["74"] = [
        LineIntrusionDetector(
            line_coords=[(10, 140), (540, 140)],
            instance_id="cam1_entrance",
            stream_id="camera_1"
        )
    ]
    
    # Camera 2: Vertical line, right-to-left only
    stream_features["73"] = [
        LineIntrusionDetector(
            line_coords=[(200, 100), (200, 380)],
            instance_id="cam2_exit",
            stream_id="camera_2"
        )
    ]

    try:
        # Start all stream capture threads (PARALLEL)
        stream_manager.start_all()
        logger.info("All streams started - parallel capture running")
        time.sleep(2)
        
        while True:
            # Get latest frames from all streams (non-blocking)
            frames = stream_manager.get_all_latest_frames()
            
            if not frames:
                time.sleep(0.001)
                continue
            
            # Process each stream's latest frame SEQUENTIALLY
            for stream_id, frame in frames.items():
                # Get features for this stream
                features = stream_features.get(stream_id, [])
                
                # Process: DETECT ONCE → FEATURES CLASSIFY
                annotated_frame, events = frame_processor.process_frame(
                    frame, 
                    stream_id, 
                    features
                )
                
                # Update display
                stream_manager.update_annotated_frame(stream_id, annotated_frame)
                
                # Log events
                if events:
                    for event in events:
                        logger.warning(
                            f"[EVENT] {event['stream_id']}: "
                            f"Track {event['track_id']} crossed {event['direction']} "
                            f"(instance: {event['instance_id']})"
                        )
            
            # Display all streams
            if enable_display:
                if not stream_manager.display_all_streams():
                    logger.info("Display quit requested")
                    break
            
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        stream_manager.stop_all()
        logger.info("System shutdown complete")


if __name__ == "__main__":
    main()
