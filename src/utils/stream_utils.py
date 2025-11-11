# utils/stream_utils.py
from src.utils.utils import start_daemon_thread, safe_close_queue
from src.utils.live_visualization import LiveVisualization
from src.notifications.alert_system import AlertSystem
from src.pipelines.forwarder_worker import ForwarderWorker

def create_stream_components(stream_id, stream_cfg, width, height):
    """Create all per-stream components."""
    alert_sys = AlertSystem(snapshot_dir=f"snapshots_{stream_id}")
    visualizer = LiveVisualization(f"Stream {stream_id}")
    dims = (
        int(stream_cfg.get("camera_width", width)),
        int(stream_cfg.get("camera_height", height))
    )
    return alert_sys, visualizer, dims


def start_forwarder_thread(
    sid,
    in_queue,
    out_queue,
    stream_cfg,
    store,
    stop_event,
    threads_list,
    logger,
    is_stream_active_fn,
    within_time_bounds_fn,
):
    """Initialize and start a ForwarderWorker thread for a stream."""
    fw = ForwarderWorker(
        stream_id=sid,
        in_queue=in_queue,
        out_queue=out_queue,
        stream_cfg=stream_cfg,
        is_active_fn=is_stream_active_fn,
        within_time_bounds_fn=within_time_bounds_fn,
        stop_event=stop_event,
        instance_id=store.get_instance_id(sid),
    )
    t = start_daemon_thread(fw.run, name=f"forwarder-{sid}", threads_list=threads_list)
    logger.debug(f"Started forwarder for stream {sid}")
    return t
