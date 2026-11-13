"""Local run evidence; never infer receiver success from a UDP send."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path


def read_unity_snapshot(path, session_id, *, not_before_unix_ms, now_unix_ms=None,
                        expected_udp_port=None, sent_frames=None):
    """Accept only fresh, schema-checked snapshots for this exact producer session."""
    unavailable = {"status": "unavailable", "reason": "no_matching_fresh_snapshot"}
    if session_id is None:
        return {**unavailable, "reason": "producer_session_unknown"}
    try:
        path = Path(path)
        if path.stat().st_size > 65536:
            return {**unavailable, "reason": "invalid_snapshot"}
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        now = time.time() * 1000 if now_unix_ms is None else now_unix_ms
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
        ):
            raise ValueError("schema")
        if data.get("session_id") != str(session_id):
            return unavailable
        stamp = data["written_unix_ms"]
        if type(stamp) not in (int, float) or not math.isfinite(stamp):
            raise ValueError("timestamp")
        if stamp < not_before_unix_ms or not -1000 <= now - stamp <= 2000:
            return {**unavailable, "reason": "stale_snapshot"}
        for key in (
            "received_pose_frames",
            "applied_pose_frames",
            "rejected_packets_receiver_total",
            "rendered_frames",
            "udp_port",
        ):
            if type(data.get(key)) is not int or data[key] < 0:
                raise ValueError(key)
        if not 1 <= data["udp_port"] <= 65535:
            raise ValueError("port")
        for key in ("duration_s", "working_set_mib"):
            if (
                type(data.get(key)) not in (int, float)
                or not math.isfinite(data[key])
                or data[key] < 0
            ):
                raise ValueError(key)
        if data["applied_pose_frames"] > data["received_pose_frames"]:
            raise ValueError("applied count")
        if expected_udp_port is not None and data["udp_port"] != expected_udp_port:
            return {**unavailable, "reason": "receiver_port_mismatch"}
        if sent_frames is not None and (
            type(sent_frames) is not int or sent_frames < 0
            or data["received_pose_frames"] > sent_frames
        ):
            return {**unavailable, "reason": "receiver_count_mismatch"}
        for key in ("headless", "receiving", "working_set_available"):
            if type(data.get(key)) is not bool:
                raise ValueError(key)
        # Optional additive evidence: preserve compatibility with older snapshots.
        if "target_frame_rate" in data and (
            type(data["target_frame_rate"]) is not int
            or (data["target_frame_rate"] != -1 and data["target_frame_rate"] <= 0)
        ):
            raise ValueError("target_frame_rate")
        if "v_sync_count" in data and (
            type(data["v_sync_count"]) is not int or not 0 <= data["v_sync_count"] <= 4
        ):
            raise ValueError("v_sync_count")
        if "demo_frame_limit_active" in data and (
            type(data["demo_frame_limit_active"]) is not bool
            or (data["demo_frame_limit_active"] and (
                data.get("target_frame_rate") != 60 or data.get("v_sync_count") != 0
            ))
        ):
            raise ValueError("demo_frame_limit_active")
        result = {**data, "status": "observed"}
        if not data["working_set_available"]:
            for key in ("working_set_mib", "working_set_growth_mib", "peak_working_set_mib"):
                result[key] = None
        result["render_fps"] = (
            data["rendered_frames"] / data["duration_s"]
            if not data["headless"] and data["duration_s"] > 0 and data["rendered_frames"] > 0
            else None
        )
        result["render_scope"] = (
            "built-in Game camera post-render callbacks; not screen presentation"
        )
        result["render_status"] = (
            "observed"
            if result["render_fps"] is not None and data["rendered_frames"] > 0
            else "not_assessed"
        )
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return {**unavailable, "reason": "missing_or_invalid_snapshot"}


def assemble_run_report(producer, unity, *, source, requested_duration_s):
    sent = producer.get("sent_frames")
    if sent is None:
        sent = producer.get("capture_available_to_send_complete_ms", {}).get("samples")
    observed = unity.get("status") == "observed"
    received = unity.get("received_pose_frames") if observed else None
    consistent = type(sent) is int and type(received) is int and received <= sent
    return {
        "schema_version": 1,
        "source": source,
        "requested_duration_s": requested_duration_s,
        "producer": producer,
        "unity": unity,
        "transport": {
            "sent_pose_frames": sent,
            "received_pose_frames": received,
            "unconfirmed_pose_frames": sent - received if consistent else None,
            "status": "observed" if consistent else "not_assessed",
            "scope": "session counts at snapshot; unconfirmed is not proven network loss",
        },
        "acceptance_status": "not_assessed",
        "limitations": [
            "capture-to-send is not capture-to-screen latency",
            "camera run requires a person and a separate ten-minute acceptance test",
            "no aligned ground truth: pose error and effective latency are not assessed",
        ],
    }
