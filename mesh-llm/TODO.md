# mesh-llm TODO

## Connection Stability — Studio relay decay

Studio's relay connections degrade over hours until fully isolated (~10 hours). Pattern:
- Fresh restart: connects fine, RTTs 250-400ms via relay
- Over hours: RTTs climb 400→1000→2000→5000ms
- Eventually: 0 peers, can't reconnect, Nostr rediscovery fails
- Restart fixes it immediately

Confirmed via heartbeat debug logging (v0.35.2). Not a deadlock (heartbeat runs every 60s). Not code — ephemeral test also fails when Studio is isolated. Likely relay or iroh transport issue. **All peers** (Local, Mini, Fly console, Fly API) lose Studio simultaneously.

**STUN also fails** on Studio (`STUN: could not discover public address`). Both machines share public IP `180.181.228.108` — hairpin NAT means LAN UDP works but STUN from the same public IP doesn't help.

Next steps:
- [ ] Add relay health monitoring (log relay reconnects, detect stale relay)
- [ ] Try iroh default relays alongside Fly relay (maybe Fly relay specifically degrades)
- [ ] Add periodic relay reconnect (if no peers for N minutes, force relay cycle)
- [ ] Profile iroh relay websocket — is it the relay server or the client connection dying?

## Mobile Chat App (exemplar)

Build a delightful mobile chat app that connects to any mesh as a client-only node.

- Scan a QR code (mesh invite token) to join
- Client-only: no GPU, no model serving — just routes inference through mesh hosts
- Uses iroh relay for connectivity (works through NAT, cellular, etc.)
- Minimal native UI — conversation list, chat bubbles, model picker from mesh catalog
- Target: iOS first (Swift + iroh-ffi), Android follow-up
- Shows off mesh-llm's value: scan a code, get access to a GPU pool, no setup
- Think "AirDrop for AI" — one scan and you're chatting with a 235B model
- OpenAI-compatible API underneath, so could also power shortcuts/widgets

## Smart Router
- [x] Heuristic classifier: Code/Reasoning/Chat/Creative/ToolCall categories
- [x] Complexity detection: Quick/Moderate/Deep from message signals
- [x] Task-dominant scoring: match bonus + tier + position
- [x] Tool capability filter: hard gate on `tools: bool` per model profile
- [x] needs_tools as attribute, not category override
- [ ] **Static speed estimates**: Add `tok_s: f64` to ModelProfile (known from benchmarks, no runtime measurement). Feed into scoring so Quick tasks prefer fast models.
- [ ] **Response quality checks**: Detect empty/repetitive/truncated responses, trigger retry with different model. Needs proxy to inspect response bytes (currently raw TCP relay).
- [ ] **Complexity → context budget**: Deep requests get larger `-n` (max tokens), Quick gets smaller. Currently all requests use llama-server defaults.

## Multi-Model Serving
- [x] `--model A --model B` runs separate election loops per model
- [x] Auto model packs by VRAM tier
- [x] `serving_models: Vec<String>` in gossip (backward compatible)
- [x] Router picks best model per request
- [ ] **Demand-based model upgrade**: Large-VRAM host serving a small model should upgrade when demand exists for a bigger model nobody is serving.

## First-Time Experience
- [ ] **Solo fallback — fast starter model**: When `--auto` finds no mesh, download a small starter model first (Qwen2.5-3B, 2GB, ~1 min), start serving immediately, then background-download a better model for the node's VRAM tier.
- [ ] **Uptime signal**: Add `started_at: u64` to `MeshListing`. Score bonus for longer-running meshes.

## Model Catalog
- [ ] **Draft model completeness**: GLM-4.7 and DeepSeek have no draft pairing.
- [ ] **Don't download what won't fit**: Check VRAM before downloading via `--model`.
- [ ] `mesh-llm recommend`: CLI subcommand to suggest models for your hardware.

## MoE Expert Sharding

Design: [MoE_PLAN.md](../MoE_PLAN.md) · Auto-deploy: [MoE_DEPLOY_DESIGN.md](../MoE_DEPLOY_DESIGN.md) · Validation: [MoE_SPLIT_REPORT.md](../MoE_SPLIT_REPORT.md)

- [x] Phase 1–3: Routing analysis, expert masking, mesh integration. Tested OLMoE-1B-7B over WAN.
- [ ] **Phase 4: lazy `moe-analyze`** — auto-run ranking for unknown MoE models. Currently unknown models fall through to PP.
- [ ] **Phase 5: probe-based session placement** — parked on `moe-probe` branch.
- [ ] **Phase 6: scale testing** — Mixtral 8×22B, Qwen3-235B-A22B.

## Resilience
- [x] Nostr re-discovery on peer loss
- [x] llama-server death watchdog
- [x] Multi-host load balancing
- [x] Demand-based duplicate hosting
- [x] API deadlock fix (v0.35.1) — snapshot locks independently, never hold multiple
- [x] VRAM-scaled context sizes — prevents OOM on small machines
- [ ] **Multi-node tensor split recovery**: If one split peer dies, re-split across remaining.
- [ ] **`kill_llama_server()` uses `pkill -f`**: Should kill by PID, not pattern match.

## Discovery & Publishing
- [ ] **Revisit `--publish` flag**: Bare `--publish` without `--mesh-name` is vestigial.

## Experiments
- [ ] Qwen3.5-397B-A17B across 128GB M4 Max + second machine (MoE, ~219GB Q4)
- [ ] Largest dense models across 2+ machines (Llama-3.3-70B, Qwen2.5-72B)
- [ ] MiniMax-M2.5 MoE split across Studio + second large machine
