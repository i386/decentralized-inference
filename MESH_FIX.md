# Mesh Connection Fix — March 14, 2026

## The Bug

Connections between LAN peers drop every 2-13 minutes with:
```
WARN: failed closing path err=LastOpenPath
INFO: Connection to XXXX closed: timed out
```

## Root Cause

Our custom QUIC transport config was overriding iroh's internal path management:

```rust
// BROKEN — overrides iroh's coordinated path/connection keep-alive settings
let transport_config = QuicTransportConfig::builder()
    .max_concurrent_bidi_streams(1024u32.into())
    .max_idle_timeout(Some(Duration::from_secs(30).try_into()?))
    .keep_alive_interval(Duration::from_secs(5))
    .build();
```

iroh's `QuicTransportConfigBuilder::new()` sets a coordinated set of values:
- `keep_alive_interval`: 5s (connection-level)
- `default_path_keep_alive_interval`: 5s (per-path)
- `default_path_max_idle_timeout`: 6.5s (per-path)
- `max_idle_timeout`: 30s (quinn default, connection-level)

When we called `.max_idle_timeout()` and `.keep_alive_interval()`, we were replacing
the builder's defaults. This interfered with iroh's multipath system — paths would
die, iroh would try to close the last path ("LastOpenPath"), and ~24s later the
connection would time out.

## The Fix

```rust
// FIXED — only override what we actually need, let iroh manage keep-alive/timeouts
let transport_config = QuicTransportConfig::builder()
    .max_concurrent_bidi_streams(1024u32.into())
    .build();
```

The 1024 bidi stream limit is needed for llama.cpp RPC tensor transfers.

## What We Proved

| Test | Config | Result |
|---|---|---|
| Custom timeout+keepalive, ephemeral keys, named mesh | Old | Dies every 2 min |
| Custom timeout+keepalive, signed binary, Fly relay+defaults | Old | Dies every 8-13 min |
| **iroh defaults, signed binary, Local↔Studio** | **Fix** | **10+ min stable (Studio went offline due to VPN)** |
| **iroh defaults, HEAD build, Local↔Mini** | **Fix** | **15+ min stable, zero drops** |
| **v0.35.0, mainnet, Local+Studio+Mini+2×Fly** | **Fix** | **15+ min all connected, Studio flapped once then recovered** |

## Other Issues Found & Fixed

### Binary Signing
macOS managed firewall re-prompts on every new unsigned binary (different CDHash).
Sign with Developer ID before SCP:
```bash
codesign -s "Developer ID Application: Michael Neale (W2L75AE9HQ)" -f mesh-llm
```

### Probe Removal
`probe_mesh_health()` creates a separate ephemeral endpoint to test connectivity.
This always fails for firewalled peers (even though the real `node.join()` works
via relay). Removed from both initial `--auto` join and re-discovery paths.

### dispatch_streams peer removal
When `dispatch_streams` connection dies and outbound reconnect fails, remove the
peer immediately (v0.27 behavior). Don't "keep for heartbeat retry" — the heartbeat
reconnect also fails for firewalled peers, causing death broadcast storms.

## Observations

- **LAN UDP works fine** between Local and Studio (verified with netcat test)
- **Fly clients are rock solid** — relay connections never drop
- **The mesh Rust code (dispatch_streams, heartbeat) is unchanged since v0.27** which was stable
- **STUN sometimes fails** on Studio ("could not discover public address") — managed firewall may block STUN responses intermittently
- **iroh picks closest relay as home** — Asia-Pacific canary relay wins from Australia even when our US West relay is configured

## Deploy Checklist Additions

1. Sign binary: `codesign -s "Developer ID Application: Michael Neale (W2L75AE9HQ)" -f mesh-llm`
2. SCP to remote
3. No need for `xattr -d com.apple.quarantine` if properly signed
4. No firewall prompt expected for signed binaries
