//! HTTP proxy plumbing — request parsing, model routing, response helpers.
//!
//! Used by the API proxy (port 9337), bootstrap proxy, and passive mode.
//! All inference traffic flows through these functions.

use crate::{election, mesh, router, tunnel};
use anyhow::Result;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

// ── Request parsing ──

/// Peek at an HTTP request without consuming it. Returns bytes peeked and optional model name.
pub async fn peek_request(stream: &TcpStream, buf: &mut [u8]) -> Result<(usize, Option<String>)> {
    let n = stream.peek(buf).await?;
    if n == 0 {
        anyhow::bail!("Empty request");
    }
    let model = extract_model_from_http(&buf[..n]);
    Ok((n, model))
}

fn split_http_request(buf: &[u8]) -> Option<(&str, &str)> {
    let s = std::str::from_utf8(buf).ok()?;
    let body_start = s.find("\r\n\r\n")?;
    Some((&s[..body_start], &s[body_start + 4..]))
}

fn request_line(buf: &[u8]) -> Option<&str> {
    let (headers, _) = split_http_request(buf)?;
    headers.lines().next()
}

fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines()
        .skip(1)
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then_some(value.trim())
        })
}

fn extract_json_string_field(body: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\"");
    let pos = body.find(&key)?;
    let after_key = &body[pos + key.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_ws = after_colon.trim_start();
    let after_quote = after_ws.strip_prefix('"')?;
    let end = after_quote.find('"')?;
    Some(after_quote[..end].to_string())
}

fn multipart_boundary(content_type: &str) -> Option<&str> {
    content_type
        .split(';')
        .map(str::trim)
        .find_map(|part| part.strip_prefix("boundary=").map(|value| value.trim_matches('"')))
}

fn extract_multipart_field(body: &str, boundary: &str, field: &str) -> Option<String> {
    let marker = format!("name=\"{field}\"");
    let boundary_marker = format!("--{boundary}");
    for part in body.split(&boundary_marker) {
        if !part.contains(&marker) {
            continue;
        }
        let body_start = part.find("\r\n\r\n")? + 4;
        let value = part[body_start..]
            .trim_matches(|c| c == '\r' || c == '\n')
            .trim();
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

/// Return the HTTP request path, without any query string.
pub fn request_path(buf: &[u8]) -> Option<String> {
    let line = request_line(buf)?;
    let mut parts = line.split_whitespace();
    let _method = parts.next()?;
    let raw_path = parts.next()?;
    Some(raw_path.split('?').next()?.to_string())
}

/// Extract `"model"` from a JSON or multipart OpenAI-style request body.
pub fn extract_model_from_http(buf: &[u8]) -> Option<String> {
    let (headers, body) = split_http_request(buf)?;
    if let Some(ct) = header_value(headers, "content-type") {
        if ct.starts_with("multipart/form-data") {
            let boundary = multipart_boundary(ct)?;
            return extract_multipart_field(body, boundary, "model");
        }
    }
    extract_json_string_field(body, "model")
}

/// Extract a session hint from an HTTP request for MoE sticky routing.
/// Looks for "user" or "session_id" in the JSON body. Falls back to None.
pub fn extract_session_hint(buf: &[u8]) -> Option<String> {
    let (_, body) = split_http_request(buf)?;
    // Try "user" field first (standard OpenAI parameter)
    for key in &["user", "session_id"] {
        if let Some(value) = extract_json_string_field(body, key) {
            return Some(value);
        }
    }
    None
}

/// Try to parse the JSON body from a peeked HTTP request buffer.
pub fn extract_body_json(buf: &[u8]) -> Option<serde_json::Value> {
    let (_, body) = split_http_request(buf)?;
    serde_json::from_str(body).ok()
}

pub fn is_models_list_request(buf: &[u8]) -> bool {
    matches!(
        (request_line(buf), request_path(buf).as_deref()),
        (Some(line), Some("/v1/models" | "/models")) if line.starts_with("GET ")
    )
}

pub fn is_drop_request(buf: &[u8]) -> bool {
    matches!(
        (request_line(buf), request_path(buf).as_deref()),
        (Some(line), Some("/mesh/drop")) if line.starts_with("POST ")
    )
}

pub fn is_chat_completions_request(buf: &[u8]) -> bool {
    matches!(
        (request_line(buf), request_path(buf).as_deref()),
        (Some(line), Some("/v1/chat/completions")) if line.starts_with("POST ")
    )
}

// ── Model-aware tunnel routing ──

/// The common request-handling path used by idle proxy, passive proxy, and bootstrap proxy.
///
/// Peeks at the HTTP request, handles `/v1/models`, resolves the target host
/// by model name (or falls back to any host), and tunnels the request via QUIC.
///
/// Set `track_demand` to record requests for demand-based rebalancing.
pub async fn handle_mesh_request(node: mesh::Node, tcp_stream: TcpStream, track_demand: bool) {
    let mut buf = vec![0u8; 32768];
    let (n, model_name) = match peek_request(&tcp_stream, &mut buf).await {
        Ok(v) => v,
        Err(_) => return,
    };

    // Handle /v1/models
    if is_models_list_request(&buf[..n]) {
        let served = node.models_being_served().await;
        let _ = send_models_list(tcp_stream, &served).await;
        return;
    }

    // Demand tracking for rebalancing (done after routing so we track the actual model used)
    // We'll track below after routing resolves the effective model

    // Smart routing: if no model specified (or model="auto"), classify and pick
    let routed_model = if model_name.is_none() || model_name.as_deref() == Some("auto") {
        if let Some(body_json) = extract_body_json(&buf[..n]) {
            let cl = router::classify(&body_json);
            let served = node.models_being_served().await;
            let available: Vec<(&str, f64)> = served.iter()
                .map(|name| (name.as_str(), 0.0))
                .collect();
            let picked = router::pick_model_classified(&cl, &available);
            if let Some(name) = picked {
                tracing::info!("router: {:?}/{:?} tools={} → {name}", cl.category, cl.complexity, cl.needs_tools);
                Some(name.to_string())
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    let effective_model = routed_model.or(model_name);

    // Demand tracking for rebalancing
    if track_demand {
        if let Some(ref name) = effective_model {
            node.record_request(name);
        }
    }

    // Resolve target hosts by model name, fall back to any host
    let target_hosts = if let Some(ref name) = effective_model {
        node.hosts_for_model(name).await
    } else {
        vec![]
    };
    let target_hosts = if target_hosts.is_empty() {
        match node.any_host().await {
            Some(p) => vec![p.id],
            None => {
                let _ = send_503(tcp_stream).await;
                return;
            }
        }
    } else {
        target_hosts
    };

    // Try each host in order — if tunnel fails, retry with next.
    // On first failure, trigger background gossip refresh so future requests
    // have a fresh routing table (doesn't block the retry loop).
    let mut last_err = None;
    let mut refreshed = false;
    for target_host in &target_hosts {
        match node.open_http_tunnel(*target_host).await {
            Ok((quic_send, quic_recv)) => {
                if let Err(e) = tunnel::relay_tcp_via_quic(tcp_stream, quic_send, quic_recv).await {
                    tracing::debug!("HTTP tunnel relay ended: {e}");
                }
                return;
            }
            Err(e) => {
                tracing::warn!("Failed to tunnel to host {}: {e}, trying next", target_host.fmt_short());
                last_err = Some(e);
                // Background refresh on first failure — non-blocking
                if !refreshed {
                    let refresh_node = node.clone();
                    tokio::spawn(async move { refresh_node.gossip_one_peer().await; });
                    refreshed = true;
                }
            }
        }
    }
    // All hosts failed
    if let Some(e) = last_err {
        tracing::warn!("All hosts failed for model {:?}: {e}", effective_model);
    }
    let _ = send_503(tcp_stream).await;
}

/// Route a request to a known inference target (local llama-server or remote host).
///
/// Used by the API proxy after election has determined the target.
pub async fn route_to_target(node: mesh::Node, tcp_stream: TcpStream, target: election::InferenceTarget) {
    match target {
        election::InferenceTarget::Local(port) | election::InferenceTarget::MoeLocal(port) => {
            match TcpStream::connect(format!("127.0.0.1:{port}")).await {
                Ok(upstream) => {
                    let _inflight = node.begin_inflight_request();
                    let _ = upstream.set_nodelay(true);
                    if let Err(e) = tunnel::relay_tcp_streams(tcp_stream, upstream).await {
                        tracing::debug!("API proxy (local) ended: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("API proxy: can't reach llama-server on {port}: {e}");
                    let _ = send_503(tcp_stream).await;
                }
            }
        }
        election::InferenceTarget::Remote(host_id) | election::InferenceTarget::MoeRemote(host_id) => {
            match node.open_http_tunnel(host_id).await {
                Ok((quic_send, quic_recv)) => {
                    if let Err(e) = tunnel::relay_tcp_via_quic(tcp_stream, quic_send, quic_recv).await {
                        tracing::debug!("API proxy (remote) ended: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("API proxy: can't tunnel to host {}: {e}", host_id.fmt_short());
                    let _ = send_503(tcp_stream).await;
                }
            }
        }
        election::InferenceTarget::None => {
            let _ = send_503(tcp_stream).await;
        }
    }
}

// ── Response helpers ──

pub async fn send_models_list(mut stream: TcpStream, models: &[String]) -> std::io::Result<()> {
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            let has_vision = crate::download::MODEL_CATALOG.iter()
                .find(|c| c.name == m.as_str()
                    || c.file.strip_suffix(".gguf").unwrap_or(c.file) == m.as_str())
                .map(|c| c.mmproj.is_some())
                .unwrap_or(false);
            let mut caps = vec!["text"];
            if has_vision { caps.push("vision"); }
            serde_json::json!({
                "id": m,
                "object": "model",
                "owned_by": "mesh-llm",
                "capabilities": caps,
            })
        })
        .collect();
    let body = serde_json::json!({ "object": "list", "data": data }).to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_json_ok(mut stream: TcpStream, data: &serde_json::Value) -> std::io::Result<()> {
    let body = data.to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_400(mut stream: TcpStream, msg: &str) -> std::io::Result<()> {
    let body = format!("{{\"error\":\"{msg}\"}}");
    let resp = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_503(mut stream: TcpStream) -> std::io::Result<()> {
    let body = r#"{"error":"No inference server available — election in progress"}"#;
    let resp = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

/// Pipeline-aware HTTP proxy for local targets.
///
/// Instead of TCP tunneling, this:
/// 1. Parses the HTTP request body
/// 2. Calls the planner model for a pre-plan
/// 3. Injects the plan into the request
/// 4. Forwards to the strong model via HTTP
/// 5. Streams the response back to the client
pub async fn pipeline_proxy_local(
    mut client_stream: TcpStream,
    buf: &[u8],
    n: usize,
    planner_port: u16,
    planner_model: &str,
    strong_port: u16,
    node: &mesh::Node,
) {
    // Parse request body
    let mut body = match extract_body_json(&buf[..n]) {
        Some(b) => b,
        None => {
            let _ = send_400(client_stream, "invalid JSON body").await;
            return;
        }
    };

    // Extract whether this is a streaming request
    let is_streaming = body.get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // Pre-plan: ask the small model
    let http_client = reqwest::Client::new();
    let planner_url = format!("http://127.0.0.1:{planner_port}");
    let messages = body.get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    match crate::pipeline::pre_plan(&http_client, &planner_url, planner_model, &messages).await {
        Ok(plan) => {
            tracing::info!(
                "pipeline: pre-plan by {} in {}ms — {}",
                plan.model_used, plan.elapsed_ms,
                plan.plan_text.chars().take(200).collect::<String>()
            );
            crate::pipeline::inject_plan(&mut body, &plan);
        }
        Err(e) => {
            tracing::warn!("pipeline: pre-plan failed ({e}), proceeding without plan");
        }
    }

    // Forward to strong model — use reqwest for full HTTP handling
    let strong_url = format!("http://127.0.0.1:{strong_port}/v1/chat/completions");

    let _inflight = node.begin_inflight_request();

    if is_streaming {
        // Streaming: forward SSE chunks to client
        match http_client.post(&strong_url)
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => {
                let status = resp.status();
                let content_type = resp.headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("text/event-stream")
                    .to_string();

                // Send HTTP response headers
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n\r\n",
                );
                if client_stream.write_all(header.as_bytes()).await.is_err() {
                    return;
                }

                // Stream body chunks
                use tokio_stream::StreamExt;
                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            // HTTP chunked encoding
                            let chunk_header = format!("{:x}\r\n", bytes.len());
                            if client_stream.write_all(chunk_header.as_bytes()).await.is_err() { break; }
                            if client_stream.write_all(&bytes).await.is_err() { break; }
                            if client_stream.write_all(b"\r\n").await.is_err() { break; }
                        }
                        Err(e) => {
                            tracing::debug!("pipeline: stream error: {e}");
                            break;
                        }
                    }
                }
                // Terminal chunk
                let _ = client_stream.write_all(b"0\r\n\r\n").await;
                let _ = client_stream.shutdown().await;
            }
            Err(e) => {
                tracing::warn!("pipeline: strong model request failed: {e}");
                let _ = send_503(client_stream).await;
            }
        }
    } else {
        // Non-streaming: simple request/response
        match http_client.post(&strong_url)
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => {
                let status = resp.status();
                match resp.bytes().await {
                    Ok(resp_bytes) => {
                        let header = format!(
                            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                            resp_bytes.len()
                        );
                        let _ = client_stream.write_all(header.as_bytes()).await;
                        let _ = client_stream.write_all(&resp_bytes).await;
                        let _ = client_stream.shutdown().await;
                    }
                    Err(e) => {
                        tracing::warn!("pipeline: response read failed: {e}");
                        let _ = send_503(client_stream).await;
                    }
                }
            }
            Err(e) => {
                tracing::warn!("pipeline: strong model request failed: {e}");
                let _ = send_503(client_stream).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_session_hint_user_field() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"qwen\",\"user\":\"alice\",\"messages\":[]}";
        assert_eq!(extract_session_hint(req), Some("alice".to_string()));
    }

    #[test]
    fn test_extract_session_hint_session_id() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"qwen\",\"session_id\":\"sess-42\"}";
        assert_eq!(extract_session_hint(req), Some("sess-42".to_string()));
    }

    #[test]
    fn test_extract_session_hint_user_preferred_over_session_id() {
        // "user" appears before "session_id" in our search order
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"user\":\"bob\",\"session_id\":\"sess-1\"}";
        assert_eq!(extract_session_hint(req), Some("bob".to_string()));
    }

    #[test]
    fn test_extract_session_hint_none() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"qwen\",\"messages\":[]}";
        assert_eq!(extract_session_hint(req), None);
    }

    #[test]
    fn test_extract_session_hint_no_body() {
        let req = b"GET /v1/models HTTP/1.1\r\n\r\n";
        assert_eq!(extract_session_hint(req), None);
    }

    #[test]
    fn test_extract_session_hint_no_headers_end() {
        let req = b"POST /v1/chat body without proper headers";
        assert_eq!(extract_session_hint(req), None);
    }

    #[test]
    fn test_extract_session_hint_whitespace_variants() {
        // Extra whitespace around colon and value
        let req = b"POST / HTTP/1.1\r\n\r\n{\"user\" : \"charlie\" }";
        assert_eq!(extract_session_hint(req), Some("charlie".to_string()));
    }

    #[test]
    fn test_extract_session_hint_empty_value() {
        let req = b"POST / HTTP/1.1\r\n\r\n{\"user\":\"\"}";
        assert_eq!(extract_session_hint(req), Some("".to_string()));
    }

    #[test]
    fn test_extract_model_from_http_basic() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"Qwen3-30B\"}";
        assert_eq!(extract_model_from_http(req), Some("Qwen3-30B".to_string()));
    }

    #[test]
    fn test_extract_model_from_http_multipart() {
        let req = b"POST /v1/audio/transcriptions HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=----mesh\r\n\r\n------mesh\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n<bytes>\r\n------mesh\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n------mesh--\r\n";
        assert_eq!(extract_model_from_http(req), Some("whisper-1".to_string()));
    }

    #[test]
    fn test_request_path_strips_query_string() {
        let req = b"GET /v1/models?limit=20 HTTP/1.1\r\nHost: localhost\r\n\r\n";
        assert_eq!(request_path(req), Some("/v1/models".to_string()));
    }

    #[test]
    fn test_is_models_list_request_exact_path_only() {
        let req = b"GET /v1/models/abc HTTP/1.1\r\nHost: localhost\r\n\r\n";
        assert!(!is_models_list_request(req));
    }

    #[test]
    fn test_is_chat_completions_request() {
        let req = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{}";
        assert!(is_chat_completions_request(req));
    }
}
