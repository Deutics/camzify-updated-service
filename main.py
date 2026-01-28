"""VAF Multistream Processing - API-driven configuration"""

import time
from stream.multistream_manager import MultiStreamManager
from core.frame_processor import FrameProcessor
from features.line_intrusion_detector import LineIntrusionDetector
from api.config_fetcher import ConfigFetcher
from config.settings import MODEL_PATH
from utils_main.logger import get_logger

logger = get_logger(__name__)


def build_stream_features_from_api(configs: dict) -> tuple:
    """Convert API configs to stream sources and feature instances"""
    stream_sources = {}
    stream_features = {}
    
    for stream_id, config in configs.items():
        rtsp_url = config.get("rtsp_url")
        if not rtsp_url:
            continue
        
        stream_sources[stream_id] = rtsp_url
        
        features_config = config.get("features", {})
        line_intrusion_configs = features_config.get("line_intrusion", [])
        
        if line_intrusion_configs:
            feature_instances = []
            for feat_cfg in line_intrusion_configs:
                try:
                    # Convert direction (0=any, 1=left, 2=right)
                    direction_code = feat_cfg.get("direction_to_use", 0)
                    if direction_code == 0:
                        direction_check = {"left": True, "right": True}
                    elif direction_code == 1:
                        direction_check = {"left": True, "right": False}
                    else:
                        direction_check = {"left": False, "right": True}
                    
                    detector = LineIntrusionDetector(
                        line_coords=feat_cfg.get("line_coords"),
                        instance_id=feat_cfg.get("instance_id"),
                        stream_id=stream_id,
                        direction_to_check=direction_check
                    )
                    feature_instances.append(detector)
                    
                except Exception as e:
                    logger.error(f"Failed to create detector for stream {stream_id}: {e}")
            
            if feature_instances:
                stream_features[stream_id] = feature_instances
    
    return stream_sources, stream_features


def main():
    enable_display = True
    
    logger.info("=== VAF Multistream Processing (API Mode) ===")
    
    # Fetch configurations from API
    logger.info("Fetching configurations from API...")
    config_fetcher = ConfigFetcher()

    stream_ids = [73,74]  # Or None for all active streams
    configs = config_fetcher.fetch_line_intrusion_configs(stream_ids=stream_ids, is_active=True)
    
    if not configs:
        logger.error("No configurations fetched from API")
        return
    
    logger.info(f"Loaded configurations for {len(configs)} streams")
    
    # Build stream sources and features from API configs
    stream_sources, stream_features = build_stream_features_from_api(configs)
    
    if not stream_sources:
        logger.error("No valid stream sources configured")
        return
    
    logger.info(f"Configured {len(stream_sources)} streams with features")
    
    # Initialize components
    stream_manager = MultiStreamManager(stream_sources, enable_display=enable_display)
    
    frame_processor = FrameProcessor(
        model_path=MODEL_PATH,
        conf_thresh=0.45
    )
    
    try:
        stream_manager.start_all()
        logger.info("Stream capture threads started")
        
        time.sleep(2)
        
        logger.info("Starting inference server loop")
        
        while True:
            frames = stream_manager.get_all_latest_frames()
            
            if not frames:
                time.sleep(0.01)
                continue
            
            for stream_id, frame in frames.items():
                try:
                    features = stream_features.get(stream_id, [])
                    
                    annotated_frame, events = frame_processor.process_frame(
                        frame, 
                        stream_id, 
                        features
                    )
                    
                    stream_manager.update_annotated_frame(stream_id, annotated_frame)
                    
                    if events:
                        for event in events:
                            logger.warning(
                                f"[EVENT] {event['stream_id']}: "
                                f"Track {event['track_id']} crossed {event['direction']}"
                            )
                except Exception as e:
                    logger.error(f"Processing error for {stream_id}: {e}")
                    continue
            
            if enable_display:
                if not stream_manager.display_all_streams():
                    logger.info("Quit requested")
                    break
            
    except KeyboardInterrupt:
        logger.info("Shutdown")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        stream_manager.stop_all()
        logger.info("Stopped")


if __name__ == "__main__":
    main()
