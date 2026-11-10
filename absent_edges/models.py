"""Private local model client. Tool syntax is decoded by vLLM's official parser."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
from pathlib import Path
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse


def token_ids_digest(tokens):
    """Hash ordered token IDs; equal counts alone cannot certify rendering."""
    if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('Token IDs must be a list of nonnegative integers')
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode('ascii')).hexdigest()


@dataclass(frozen=True)
class ModelProfile:
    endpoint: str = "http://127.0.0.1:8100"
    model: str = "absent-edges-gemma4-31b-bf16"
    revision: str = "518276fb130dc81caf9a4f772e65e63ef2526493"
    max_context: int = 32768
    enable_thinking: bool = False

    def __post_init__(self):
        if type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be an explicit boolean")
        parsed = urlparse(self.endpoint)
        try:
            local = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            local = False
        if parsed.scheme != "http" or not local or parsed.username or parsed.password:
            raise ValueError("Inference endpoint must be a literal loopback HTTP address")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("Endpoint must contain only scheme, loopback address, and port")


class LocalModel:
    def __init__(self, profile: ModelProfile | None = None, timeout: float = 600):
        self.profile = profile or ModelProfile()
        self.timeout = timeout
        # No proxy can redirect private inference or disclose an episode externally.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tokenize(self, messages, tools=None):
        """Use the serving process's exact template/tokenizer, including tools."""
        request = {'model': self.profile.model, 'messages': messages,
                   'add_generation_prompt': True, 'add_special_tokens': False,
                   'chat_template_kwargs': {'enable_thinking': self.profile.enable_thinking}}
        if tools:
            request['tools'] = tools
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
        # Caller may append later messages after this call. Provenance must
        # retain the exact request sent, not a live reference to that list.
        request = json.loads(encoded)
        started = time.monotonic()
        req = urllib.request.Request(self.profile.endpoint.rstrip('/') + '/tokenize',
                                     encoded, {'Content-Type': 'application/json'})
        with self.opener.open(req, timeout=self.timeout) as response:
            raw = response.read()
        result = json.loads(raw)
        if type(result.get('count')) is not int or result['count'] < 0:
            raise ValueError('Server returned invalid exact token count')
        if result.get('max_model_len') != self.profile.max_context:
            raise ValueError('Server context identity differs from the profile')
        token_digest = token_ids_digest(result.get('tokens'))
        if len(result['tokens']) != result['count']:
            raise ValueError('Tokenization count differs from returned token IDs')
        return {'count': result['count'], 'max_model_len': result['max_model_len'],
                'token_ids_sha256': token_digest,
                'request_sha256': hashlib.sha256(encoded).hexdigest(),
                'response_sha256': hashlib.sha256(raw).hexdigest(),
                'elapsed_seconds': time.monotonic() - started}

    def chat(self, messages, tools=None, *, seed=0, max_tokens=1024, temperature=0.0):
        request = {"model": self.profile.model, "messages": messages,
                   "seed": seed, "max_tokens": max_tokens, "temperature": temperature,
                   "return_token_ids": True,
                   "chat_template_kwargs": {"enable_thinking": self.profile.enable_thinking}}
        if tools:
            request.update(tools=tools, tool_choice="auto")
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
        request = json.loads(encoded)
        started = time.monotonic()
        req = urllib.request.Request(self.profile.endpoint.rstrip("/") + "/v1/chat/completions",
                                     encoded, {"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as response:
            raw = response.read()
        result = json.loads(raw)
        if not result.get("choices") or "message" not in result["choices"][0]:
            raise ValueError("Missing parsed model message")
        return {"request": request, "response": result,
                "request_sha256": hashlib.sha256(encoded).hexdigest(),
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "elapsed_seconds": time.monotonic() - started}


TOOL_ARGUMENT_BYTES = 65536
TOOL_ARGUMENT_DEPTH = 64
TOOL_CALLS_PER_MESSAGE = 64


def _tool_arguments(raw):
    """Bound model JSON before decoding; malformed output is a format failure."""
    if not isinstance(raw, str) or len(raw) > TOOL_ARGUMENT_BYTES:
        raise ValueError('Tool arguments must be a bounded JSON string')
    try:
        if len(raw.encode('utf-8')) > TOOL_ARGUMENT_BYTES:
            raise ValueError('Tool arguments exceed the byte limit')
    except UnicodeError as error:
        raise ValueError('Tool arguments are not valid UTF-8') from error
    depth, quoted, escaped = 0, False, False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            depth += 1
            if depth > TOOL_ARGUMENT_DEPTH:
                raise ValueError('Tool arguments exceed the nesting limit')
        elif char in ']}':
            depth -= 1

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate tool argument key')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('Tool arguments contain a nonfinite number')

    def finite_float(value):
        result = float(value)
        return result if math.isfinite(result) else invalid_constant(value)

    try:
        result = json.loads(raw, object_pairs_hook=unique_object,
                            parse_constant=invalid_constant, parse_float=finite_float)
    except RecursionError as error:
        raise ValueError('Tool arguments exceed the decoder nesting limit') from error
    if not isinstance(result, dict):
        raise ValueError('Tool arguments must be an object')
    return result


def parsed_tool_calls(message, allowed_tools):
    """Validate native calls without text recovery or exception-type guessing.

    These prospective limits apply equally to all arms. The caller accounts for
    the native response first, then maps ValueError to mini's FormatError path.
    Transport, token-accounting and controller failures stay outside this scope.
    """
    if not isinstance(message, dict):
        raise ValueError('Model message must be an object')
    if message.get('role', 'assistant') != 'assistant':
        raise ValueError('Model message must have assistant role')
    if message.get('content') is not None and not isinstance(message['content'], str):
        raise ValueError('Model content must be text or null')
    calls = message.get('tool_calls')
    if calls is None:
        calls = []
    if not isinstance(calls, list) or len(calls) > TOOL_CALLS_PER_MESSAGE:
        raise ValueError('Tool calls must be a bounded list')
    result = []
    identifiers = set()
    for call in calls:
        if not isinstance(call, dict) or call.get('type') != 'function':
            raise ValueError('Malformed tool call')
        identifier = call.get('id')
        if (not isinstance(identifier, str) or not identifier or len(identifier) > 256
                or identifier in identifiers):
            raise ValueError('Tool call identifier must be unique bounded text')
        identifiers.add(identifier)
        function = call.get('function')
        if not isinstance(function, dict):
            raise ValueError('Tool function must be an object')
        name = function.get('name')
        if not isinstance(name, str) or len(name) > 128 or name not in allowed_tools:
            raise ValueError('Tool not allowlisted')
        arguments = _tool_arguments(function.get('arguments'))
        result.append({'id': identifier, 'name': name, 'arguments': arguments})
    return result


def synthetic_tool_smoke(client):
    """Real local inference, synthetic non-benchmark fixture; not research evidence."""
    with tempfile.TemporaryDirectory(prefix="absent-edges-tool-smoke-") as directory:
        fixture = Path(directory) / "fixture.txt"
        fixture.write_text("verification_code: ABSENT-EDGES-7291\n", encoding="utf-8")
        return _synthetic_tool_smoke(client, fixture)


def _synthetic_tool_smoke(client, fixture):
    tools = [{"type": "function", "function": {
        "name": "read_file", "description": "Read the named text file from the fixture.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}}}]
    messages = [{"role": "system", "content": "Use the read_file tool to read requested files. Never invent file content."},
                {"role": "user", "content": "Read fixture.txt using read_file and return its verification code."}]
    first = client.chat(messages, tools, max_tokens=256)
    message = first["response"]["choices"][0]["message"]
    calls = parsed_tool_calls(message, {"read_file"})
    if len(calls) != 1 or calls[0]["arguments"] != {"path": "fixture.txt"}:
        raise RuntimeError("Model did not make the expected parsed fixture read")
    assistant = {k: message[k] for k in ("role", "content", "tool_calls") if k in message}
    messages.append(assistant)
    # Execute the allowlisted read against a real temporary fixture after validating arguments.
    observation = fixture.read_text(encoding="utf-8")
    messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": observation})
    second = client.chat(messages, tools, max_tokens=256)
    final = second["response"]["choices"][0]["message"]
    passed = "ABSENT-EDGES-7291" in (final.get("content") or "") and not final.get("tool_calls")
    return {"synthetic_only": True, "passed": passed, "first": first,
            "tool_execution": {"name": "read_file", "path": "fixture.txt",
                               "content_sha256": hashlib.sha256(observation.encode()).hexdigest()},
            "tool_observation": observation, "second": second}
