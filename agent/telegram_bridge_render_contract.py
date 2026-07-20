"""Shared prompt contract for Telegram Bridge specialist render payloads."""

CANONICAL_RENDER_ENVELOPE_INSTRUCTIONS = (
    "Use this render envelope contract for every response. Return only a non-empty JSON array of "
    "telegram.bridge.render_payload.v1 objects. Keep schema_version, message_id, correlation_id, "
    "action, target, and render at their schema-defined levels. "
    "One text-only, non-exclusive example is: "
    '{"schema_version":"telegram.bridge.render_payload.v1",'
    '"message_id":"msg_<event_id>_result",'
    '"correlation_id":"<event_id>",'
    '"action":"send",'
    '"target":{"chat_id":123456789},'
    '"render":{"text":"User-facing response."}}. '
    "Replace the placeholders with the authoritative event and chat values. Tables and lists use "
    "supported structured blocks under render.blocks; buttons use render.buttons. Follow profile "
    "and task instructions to choose the supported presentation that fits the response. "
    "Never put chat_id or text at the payload top level. "
    "Never wrap the object in payload."
)
