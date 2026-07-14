"""Shared prompt contract for Telegram Bridge specialist render payloads."""

CANONICAL_TEXT_RENDER_INSTRUCTIONS = (
    "For a normal text response, use exactly this object shape inside the JSON array: "
    '{"schema_version":"telegram.bridge.render_payload.v1",'
    '"message_id":"msg_<event_id>_result",'
    '"correlation_id":"<event_id>",'
    '"action":"send",'
    '"target":{"chat_id":123456789},'
    '"render":{"text":"User-facing response."}}. '
    "Replace the placeholders with the authoritative event and chat values. "
    "Never put chat_id or text at the payload top level. "
    "Never wrap the object in payload."
)
