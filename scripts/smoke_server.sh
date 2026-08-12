#!/usr/bin/env bash
# Gate S1c end-to-end: exercise the HTTP surface against a RUNNING server, over the wire, with curl.
#
# Deliberately not linked against the server: an external agentic harness reaches this process
# through a socket and nothing else, so the test should too. Run scripts/serve.sh first.
#
#   bash scripts/smoke_server.sh [host:port]
set -u
BASE="${1:-127.0.0.1:8080}"
PASS=0; FAIL=0
ok(){ echo "  [PASS] $1"; PASS=$((PASS+1)); }
no(){ echo "  [FAIL] $1"; [ $# -gt 1 ] && echo "         $2"; FAIL=$((FAIL+1)); }

j(){ python3 -c "import sys,json;d=json.load(sys.stdin);print(eval('d'+sys.argv[1]))" "$1" 2>/dev/null; }

echo "== $BASE =="

# ---- health / models -------------------------------------------------------------------------
R=$(curl -sS --max-time 10 "http://$BASE/health") || R=""
[ "$(echo "$R" | j "['status']")" = "ok" ] && ok "/health" || no "/health" "$R"
MODEL=$(echo "$R" | j "['model']")

R=$(curl -sS --max-time 10 "http://$BASE/v1/models")
# -n as well as equality: with the server down both sides are empty and "" = "" is a FALSE PASS,
# which is exactly how this script once reported 1/19 with nothing listening at all.
ID=$(echo "$R" | j "['data'][0]['id']")
[ -n "$ID" ] && [ "$ID" = "$MODEL" ] && ok "/v1/models lists the model" || no "/v1/models" "$R"

curl -sS --max-time 10 "http://$BASE/metrics" | grep -q dsv4_requests_total \
  && ok "/metrics is Prometheus text" || no "/metrics"

curl -sS --max-time 10 "http://$BASE/" | grep -q "DeepSeek-V4-Flash" \
  && ok "GET / serves the web UI" || no "GET / serves the web UI"

# ---- non-streaming chat, greedy so the answer is checkable --------------------------------------
R=$(curl -sS --max-time 600 "http://$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}],
  "thinking_mode":"chat","temperature":0,"max_tokens":24}')
C=$(echo "$R" | j "['choices'][0]['message']['content']")
echo "         content: $(echo "$C" | head -c 120)"
echo "$C" | grep -qi paris && ok "chat/completions answers the factual probe" || no "chat/completions" "$R"
[ -n "$(echo "$R" | j "['usage']['total_tokens']")" ] && ok "usage is reported" || no "usage"
[ -n "$(echo "$R" | j "['timings']['tokens_per_second']")" ] \
  && ok "timings carry tok/s ($(echo "$R" | j "['timings']['tokens_per_second']"))" || no "timings"

# ---- thinking mode surfaces reasoning separately -------------------------------------------------
R=$(curl -sS --max-time 900 "http://$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Is 91 prime? Think briefly, then answer."}],
  "thinking_mode":"thinking","reasoning_effort":"low","temperature":0,"max_tokens":200}')
RC=$(echo "$R" | j "['message'] if False else d['choices'][0]['message'].get('reasoning_content','')")
[ -n "$RC" ] && ok "thinking mode returns reasoning_content ($(echo -n "$RC" | wc -c) bytes)" \
              || no "thinking mode returns reasoning_content" "$R"
echo "$R" | grep -q '"content"' && ok "content is separate from reasoning" || no "content separate"

# ---- prefix cache: the same conversation twice, second must report cached tokens -----------------
BODY='{"messages":[{"role":"user","content":"Name three prime numbers."}],"thinking_mode":"chat","temperature":0,"max_tokens":32}'
curl -sS --max-time 600 "http://$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" >/dev/null
R=$(curl -sS --max-time 600 "http://$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY")
CT=$(echo "$R" | j "['usage']['prompt_tokens_details']['cached_tokens']")
[ "${CT:-0}" -gt 0 ] 2>/dev/null && ok "prefix cache hit on a repeated prompt ($CT tokens)" \
                                 || no "prefix cache hit on a repeated prompt" "cached=$CT"

# ---- streaming --------------------------------------------------------------------------------
S=$(curl -sS --max-time 600 -N "http://$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Count from 1 to 5."}],
  "thinking_mode":"chat","temperature":0,"max_tokens":48,"stream":true}')
echo "$S" | grep -q '^data: ' && ok "stream emits SSE data frames" || no "stream emits SSE frames"
echo "$S" | grep -q 'chat.completion.chunk' && ok "stream chunks have the right object type" || no "chunk object type"
echo "$S" | grep -q '\[DONE\]' && ok "stream terminates with [DONE]" || no "stream [DONE]"
echo "$S" | grep -q '"finish_reason": *"stop"' && ok "stream carries finish_reason" || no "finish_reason"
echo "$S" | grep -q '"usage"' && ok "stream carries a final usage chunk" || no "final usage chunk"

# ---- tools: the model must be able to emit a DSML call, parsed back to OpenAI shape ---------------
R=$(curl -sS --max-time 900 "http://$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"What is the weather in Beijing? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},
    "required":["city"]}}}],
  "thinking_mode":"chat","temperature":0,"max_tokens":256}')
if echo "$R" | grep -q '"tool_calls"'; then
  NAME=$(echo "$R" | j "['choices'][0]['message']['tool_calls'][0]['function']['name']")
  ARGS=$(echo "$R" | j "['choices'][0]['message']['tool_calls'][0]['function']['arguments']")
  ok "tool call parsed: $NAME $ARGS"
  [ "$(echo "$R" | j "['choices'][0]['finish_reason']")" = "tool_calls" ] \
    && ok "finish_reason is tool_calls" || no "finish_reason is tool_calls"
else
  # Not a server bug if the model chose prose; report it as such rather than as a pass.
  no "model emitted a tool call" "$(echo "$R" | j "['choices'][0]['message']['content']" | head -c 200)"
fi

# ---- /v1/completions ----------------------------------------------------------------------------
R=$(curl -sS --max-time 600 "http://$BASE/v1/completions" -H 'Content-Type: application/json' \
    -d '{"prompt":"The capital of France is","max_tokens":8,"temperature":0}')
T=$(echo "$R" | j "['choices'][0]['text']")
echo "         text: $(echo "$T" | head -c 80)"
echo "$T" | grep -qi paris && ok "/v1/completions raw prompt" || no "/v1/completions" "$R"

# ---- error handling -----------------------------------------------------------------------------
[ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "http://$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d '{bad json')" = "400" ] \
  && ok "malformed JSON -> 400" || no "malformed JSON -> 400"
[ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "http://$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":9999999}')" = "400" ] \
  && ok "over-long request -> 400, not a crash" || no "over-long request -> 400"

echo
echo "Gate SERVER: $PASS passed, $FAIL failed -> $([ $FAIL -eq 0 ] && echo PASS || echo FAIL)"
exit $([ $FAIL -eq 0 ] && echo 0 || echo 1)
