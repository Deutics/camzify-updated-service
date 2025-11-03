# api/routes/service_routes.py
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from api.service_controller import service_controller
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/service", tags=["Service Controls"])


# ----------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------
class StreamConfig(BaseModel):
    """Basic per-stream configuration structure."""
    stream_id: str = Field(..., example="73")
    rtsp_url: Optional[str] = Field(None, example="rtsp://user:pass@192.168.0.10:554/stream")
    https_url: Optional[str] = Field(None, example="https://media.example/73/index.m3u8")
    camera_width: Optional[int] = Field(None, example=640)
    camera_height: Optional[int] = Field(None, example=360)
    # Other custom parameters (fps, encoding, etc.) are allowed implicitly.


class RulesPayload(BaseModel):
    """Used for full rule replacement."""
    rules: Dict[str, Any]


# ----------------------------------------------------------------------
# Internal helper
# ----------------------------------------------------------------------
def _get_service():
    svc = getattr(service_controller, "service_instance", None)
    if svc is None:
        raise HTTPException(status_code=400, detail="Service not running. Start service via /service/start first.")
    return svc


# ----------------------------------------------------------------------
# Status and lifecycle endpoints
# ----------------------------------------------------------------------
@router.get("/status", summary="Service status")
async def get_status():
    """Return running status and active stream IDs."""
    svc = getattr(service_controller, "service_instance", None)
    if not svc:
        return {"running": False, "streams": []}

    return {
        "running": svc.is_running(),
        "streams": list(getattr(svc, "stream_handlers", {}).keys())
    }


@router.post("/start", summary="Start service")
async def start_service():
    """Start the multi-stream video analytics service."""
    try:
        await service_controller.start_service()
        return {"success": True, "running": True}
    except Exception as e:
        logger.exception("Failed to start service")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", summary="Stop service")
async def stop_service():
    """Stop the analytics service and cleanup."""
    try:
        await service_controller.stop_service()
        return {"success": True, "running": False}
    except Exception as e:
        logger.exception("Failed to stop service")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Stream endpoints (CRUD)
# ----------------------------------------------------------------------
@router.post("/streams", summary="Add new stream", status_code=status.HTTP_201_CREATED)
async def add_stream(cfg: StreamConfig):
    svc = _get_service()
    store = svc.rule_config_store
    sid = str(cfg.stream_id)

    if store.get_stream(sid):
        raise HTTPException(status_code=409, detail=f"Stream {sid} already exists")

    def _mutate(streams_by_id, rules_by_id):
        streams_by_id[sid] = {
            "stream_id": sid,
            "rtsp_stream_url": cfg.rtsp_url,
            "https_stream_url": cfg.https_url,
            "camera_width": cfg.camera_width,
            "camera_height": cfg.camera_height,
        }

    try:
        store.apply_mutation(_mutate)
        logger.info(f"Added stream {sid}")
        return {"success": True, "stream_id": sid}
    except Exception as e:
        logger.exception(f"Failed to add stream {sid}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/streams/{stream_id}", summary="Update existing stream")
async def update_stream(stream_id: str, cfg: StreamConfig):
    svc = _get_service()
    store = svc.rule_config_store
    sid = str(stream_id)

    if not store.get_stream(sid):
        raise HTTPException(status_code=404, detail=f"Stream {sid} not found")

    def _mutate(streams_by_id, rules_by_id):
        cur = dict(streams_by_id.get(sid, {}))
        if cfg.rtsp_url is not None:
            cur["rtsp_stream_url"] = cfg.rtsp_url
        if cfg.https_url is not None:
            cur["https_stream_url"] = cfg.https_url
        if cfg.camera_width is not None:
            cur["camera_width"] = cfg.camera_width
        if cfg.camera_height is not None:
            cur["camera_height"] = cfg.camera_height
        streams_by_id[sid] = cur

    try:
        store.apply_mutation(_mutate)
        logger.info(f"Updated stream {sid}")
        return {"success": True, "stream_id": sid}
    except Exception as e:
        logger.exception(f"Failed to update stream {sid}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/streams/{stream_id}", summary="Remove stream", status_code=status.HTTP_204_NO_CONTENT)
async def remove_stream(stream_id: str):
    svc = _get_service()
    store = svc.rule_config_store
    sid = str(stream_id)

    if not store.get_stream(sid):
        raise HTTPException(status_code=404, detail=f"Stream {sid} not found")

    def _mutate(streams_by_id, rules_by_id):
        streams_by_id.pop(sid, None)
        rules_by_id.pop(sid, None)

    try:
        store.apply_mutation(_mutate)
        logger.info(f"Removed stream {sid}")
        return None
    except Exception as e:
        logger.exception(f"Failed to remove stream {sid}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Rules endpoints (GET + PUT)
# ----------------------------------------------------------------------
@router.get("/streams/{stream_id}/rules", summary="Get rules for a stream")
async def get_rules(stream_id: str):
    svc = _get_service()
    rules = svc.rule_config_store.get_active_features(str(stream_id)) or {}
    return {"stream_id": stream_id, "rules": rules}


@router.put("/streams/{stream_id}/rules", summary="Replace all rules for a stream")
async def update_rules(stream_id: str, payload: RulesPayload):
    svc = _get_service()
    store = svc.rule_config_store
    sid = str(stream_id)

    def _mutate(streams_by_id, rules_by_id):
        rules_by_id[sid] = payload.rules or {}

    try:
        store.apply_mutation(_mutate)
        logger.info(f"Updated rules for stream {sid}")
        return {"success": True, "stream_id": sid}
    except Exception as e:
        logger.exception(f"Failed to update rules for {sid}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Config inspection and partial update
# ----------------------------------------------------------------------
@router.get("/configs", summary="Get all configs")
async def get_all_configs(normalized: bool = True):
    """
    Return all configurations.
    Use `?normalized=false` to get pixel coordinates.
    """
    svc = _get_service()
    loader = getattr(svc, "config_loader", None)
    if loader is None:
        raise HTTPException(status_code=500, detail="Config loader not available")

    try:
        cfgs = loader.load_configs()
        if not normalized:
            cfgs = loader.denormalize_all(cfgs)
        return {"success": True, "configs": cfgs}
    except Exception as e:
        logger.exception("Failed to get configs")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configs/update", summary="Partial update of configurations")
async def update_configs(request: Request):
    """
    Dynamically update parts of configuration (rules, points, precision, thresholds, etc.)
    Example payload:
    {
        "stream_id": "73",
        "updates": {
            "line_intrusion.points": [[0.2, 0.3], [0.8, 0.7]],
            "line_intrusion.precision_factor": 0.5
        }
    }
    """
    svc = _get_service()
    loader = getattr(svc, "config_loader", None)
    if loader is None:
        raise HTTPException(status_code=500, detail="Config loader not available")

    try:
        body = await request.json()
        stream_id = body.get("stream_id")
        updates = body.get("updates")
        if not stream_id or not isinstance(updates, dict):
            raise HTTPException(status_code=400, detail="Invalid payload")

        old_cfgs = loader.load_configs()
        new_cfgs = loader.apply_partial_update(old_cfgs, stream_id, updates)
        new_cfgs = loader.auto_normalize_if_needed(new_cfgs)  # <-- ensures coordinate safety
        loader.store.refresh(new_cfgs)

        logger.info(f"Configs updated for stream {stream_id}: {list(updates.keys())}")
        return {"success": True, "updated_stream": stream_id}

    except Exception as e:
        logger.exception("Failed to update configs")
        raise HTTPException(status_code=500, detail=str(e))
