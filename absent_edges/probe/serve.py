"""Exp A serving: one loopback vLLM endpoint per Core1 GPU (docs/EXPA_DESIGN.md section 7,
docs/EXPA_DESIGN_V2_DELTA.md, docs/EXPA_CORE1_RUNTIME.md, docs/EXPA_RUNBOOK.md).

Launched only by the coordinator. The vLLM flags follow
staged/core1-thinking-study-20261031a/source/common.py (served model name,
bfloat16, loopback host, max-model-len 32768, --language-model-only,
--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4,
--generation-config vllm) plus --enable-prefix-caching, exactly as verified by
the serving smoke (reports/expA-serve-smoke-20261101). Thinking is disabled per
request by the harness.

Every path is a ServeSpec parameter. The defaults are the public read-only
mirror (/mnt/kldata/absent-edges-expA-public-20261101), because Core1's
thilak.arjun (UID 1008) cannot read the private 0700 project directory
(validity review finding F3).

* ``plan`` builds the vLLM argv, an allowlisted environment with TMPDIR, HOME
  and every cache under /dev/shm/ae-expA-<run>/serve/<instance>/, and a
  hardened ``sudo systemd-run`` transient system unit that runs as
  User=thilak.arjun (printed, never executed here).
* ``preflight`` refuses: any identity other than thilak.arjun with its pinned
  UID on the host (UID 1006 on ai-core-team-1 explicitly, another person's
  account there); GPU5 by UUID, nvidia-smi index or minor number; a GPU with
  more than 128 MiB used or any process; a busy port; a cache root that is not
  local tmpfs; a serve env without bin/python or bin/ninja; and model files that
  are unreadable or do not match the model profile's manifest (models/COPY.sha256
  for the default 31B), whose own sha256 is pinned. It records the uid, user name
  and ``findmnt -no OPTIONS /proc``.
* Model profiles (``MODEL_PROFILES``, protocols/EXPA_REPL12B.md): ``gemma4-31b``
  (the default, the frozen Exp A model, unchanged), ``gemma4-12b`` and
  ``gemma4-31b-qat`` (the main-track W4A16 checkpoint). A profile sets the served
  alias, the revision, the pinned copy and its required files; the flags, dtype
  and max_model_len are the same for every profile. A profile may also pin a
  chat-template override (path and sha256): vLLM then gets ``--chat-template``,
  preflight refuses a template that is absent or has another sha256, and the plan
  records the override. Without one (31B, 12B) nothing changes.
* ``Supervisor`` (the unit's entry point) launches vLLM in its own process
  group, waits for /health and /v1/models, then every 30 s observes the GPU.
  A GPU PID that is not provably ours (a listed descendant or a member of our
  process group) is re-observed once; if it is still listed and still unproven
  it is foreign (finding F4). Hidden or vanished /proc entries are never treated
  as ours (fail closed). On a foreign PID, on GPU5, or when the GPU can no
  longer be observed, the supervisor writes a collision record, stops only its
  own process group and exits, which ends only its own unit. It never signals
  any other process.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import socket
import stat
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET

from .ledger import process_identity, require_local_filesystem, write_immutable_json

SERVE_PROTOCOL = "expA-serve-v2"
PUBLIC_ROOT = Path("/mnt/kldata/absent-edges-expA-public-20261101")
SERVE_ENV = PUBLIC_ROOT / "envs/serve"
RUNTIME_ROOT = PUBLIC_ROOT / "runtime/python3.11"
RUNTIME_PYTHON = RUNTIME_ROOT / "bin/python3.11"
MODEL_DIR = PUBLIC_ROOT / "models/gemma4-31b-bf16-518276fb"
MODEL_MANIFEST = PUBLIC_ROOT / "models/COPY.sha256"
# sha256 of models/COPY.sha256 (byte-identical to ORIGINAL.sha256), checked on Core1 2026-11-01.
MODEL_MANIFEST_SHA256 = "7e67af6d6d633a7469c2463156f936c5a38979a7a3c61e662ce85abe0211987a"
MODEL_ALLOWED_EXTRA = frozenset({".gitattributes"})  # `sha256sum *` skips dotfiles; vLLM never reads it
MODEL_REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                        "generation_config.json", "model.safetensors.index.json")
MODEL_ALIAS = "absent-edges-gemma4-31b-bf16"
MODEL_REVISION = "518276fb130dc81caf9a4f772e65e63ef2526493"
# The frozen 31B BF16 chat template (pinned by models/COPY.sha256; the 12B copy's is byte-identical). A profile whose
# own copy ships another template can be served with this one instead (ServedModel.chat_template).
MODEL_CHAT_TEMPLATE = MODEL_DIR / "chat_template.jinja"
MODEL_CHAT_TEMPLATE_SHA256 = "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0"
# Model profiles (protocols/EXPA_REPL12B.md section 2). The default, gemma4-31b, is the frozen Exp A model, built from
# the constants above unchanged. Every profile is served with the same flags, dtype and max_model_len; a profile's
# chat-template override only adds --chat-template.
DEFAULT_MODEL_PROFILE = "gemma4-31b"
A8_MODELS = "/mnt/kldata/arjun/absent-edges-a8/models"
OLMO3_ROPE_OVERRIDE = ('{"rope_parameters":{"rope_type":"yarn","attention_factor":1.2079441541679836,"beta_fast":32,'
                       '"beta_slow":1,"factor":8.0,"original_max_position_embeddings":8192,"rope_theta":500000}}')
MODEL_COMMON_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                      "generation_config.json")
# The file vLLM loads the weights through: the index of a sharded copy, or the one file of a single-file copy.
WEIGHTS_ENTRY_FILE = {"sharded": "model.safetensors.index.json", "single": "model.safetensors"}
MAX_MODEL_LEN = 32768
DEFAULT_MAX_NUM_SEQS = 8
DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
EXPECTED_USER = "thilak.arjun"
# The UID mapping is swapped between the hosts (docs/EXPA_CORE1_RUNTIME.md).
EXPECTED_UID_BY_HOST = {"ai-core-team-1": 1008, "ai-core-team-2": 1006}
# Core1 UID 1006 is another person's account; Core2 UID 1006 is ours.
FORBIDDEN_UIDS_BY_HOST = {"ai-core-team-1": frozenset({1006})}
GPU5_UUID = "GPU-7b8d9f89-0a8f-2d07-4833-bee70a0eeed2"
FORBIDDEN_GPU_UUIDS = frozenset({GPU5_UUID})
FORBIDDEN_GPU_MINORS = frozenset({5})
FORBIDDEN_GPU_INDICES = frozenset({5})
MAX_IDLE_USED_MIB = 128
SERVE_RUNTIME_MAX_SECONDS = 6 * 86400
WATCHDOG_SECONDS = 30
REOBSERVE_SECONDS = 2
CACHE_PARENT = Path("/dev/shm")
INACCESSIBLE_PATHS = ("/data5t", "/workspace", "/mnt/eu-ai-datasets")
SOURCE_TEMPLATE = "staged/core1-thinking-study-20261031a/source/common.py"
SMOKE_REFERENCE = "reports/expA-serve-smoke-20261101"
QUICK_HASH_LIMIT = 256 * 1024**2
_UUID = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
_MIB = re.compile(r"\s*([0-9]+)\s*MiB\s*")
_SUMS_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")
_OBSERVATION_ERRORS = (OSError, ValueError, subprocess.SubprocessError, ET.ParseError)


class ServePreflightError(RuntimeError):
    """The identity, GPU, port, cache location or model files are not safe to use."""


def _absolute(name, value):
    if not isinstance(value, str) or not value.startswith("/") or "\n" in value or "\0" in value:
        raise ValueError(f"{name} must be an absolute path")
    return value


def required_model_files(weights):
    """The files a model manifest must list for a weights layout: the common files plus the layout's entry file.

    ``sharded`` requires model.safetensors.index.json (the shards are whatever the pinned manifest lists);
    ``single`` requires model.safetensors and no index.
    """
    if weights not in WEIGHTS_ENTRY_FILE:
        raise ValueError(f"weights layout must be one of {', '.join(sorted(WEIGHTS_ENTRY_FILE))}, not {weights!r}")
    return MODEL_COMMON_FILES + (WEIGHTS_ENTRY_FILE[weights],)


@dataclass(frozen=True)
class ServedModel:
    """One model profile: the Hub identity, the served alias and the pinned public-mirror copy.

    ``chat_template`` and ``chat_template_sha256`` (both or neither) pin a chat-template override: vLLM serves the
    profile with ``--chat-template chat_template`` instead of the copy's own chat_template.jinja, and preflight refuses
    unless that file's sha256 is ``chat_template_sha256``. Profiles without one serve exactly as before.
    """
    name: str
    hub_id: str
    revision: str
    alias: str
    model_dir: str
    manifest: str
    manifest_sha256: str
    weights: str
    chat_template: str = None
    chat_template_sha256: str = None
    # Follow-up A8 (protocols/EXPA_A8_MODELS.md): another model family is served with its own vLLM tool-call parser,
    # its own reasoning parser (None: no --reasoning-parser) and, if needed, without prefix caching.
    # The defaults are the frozen Gemma 4 flags, so every earlier profile serves exactly as before.
    tool_call_parser: str = "gemma4"
    reasoning_parser: str = "gemma4"
    prefix_caching: bool = True
    # A JSON object vLLM applies to the Hugging Face config (--hf-overrides); None for every Gemma profile.
    hf_overrides: str = None
    # Further vLLM flags a profile needs (tensor parallelism, a native tokenizer or weight format); () for Gemma.
    extra_args: tuple = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError("a model profile name must match [a-z0-9][a-z0-9-]{0,47}")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("a model profile revision must be a full 40-hex commit")
        if not isinstance(self.manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise ValueError("a model profile pins its manifest sha256")
        for name in ("model_dir", "manifest"):
            _absolute(name, getattr(self, name))
        required_model_files(self.weights)
        if (self.chat_template is None) != (self.chat_template_sha256 is None):
            raise ValueError("a chat-template override names its path and pins its sha256 (both or neither)")
        if self.chat_template is not None:
            _absolute("chat_template", self.chat_template)
            if not isinstance(self.chat_template_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                                                 self.chat_template_sha256):
                raise ValueError("a chat-template override pins its sha256 (64 lowercase hex digits)")

    @property
    def required_files(self):
        return required_model_files(self.weights)

    @property
    def chat_template_override(self):
        """{path, sha256} of the chat-template override, or None (the copy's own template is served)."""
        if self.chat_template is None:
            return None
        return {"path": self.chat_template, "sha256": self.chat_template_sha256}

    def describe(self):
        """The profile fields a serve plan records next to path, alias, revision and manifest (the chat-template
        override only when the profile has one)."""
        record = {"profile": self.name, "hub_id": self.hub_id, "weights": self.weights,
                  "required_files": list(self.required_files)}
        if self.chat_template is not None:
            record["chat_template_override"] = self.chat_template_override
        return record


MODEL_PROFILES = {profile.name: profile for profile in (
    ServedModel(name=DEFAULT_MODEL_PROFILE, hub_id="google/gemma-4-31B-it", revision=MODEL_REVISION, alias=MODEL_ALIAS,
                model_dir=str(MODEL_DIR), manifest=str(MODEL_MANIFEST), manifest_sha256=MODEL_MANIFEST_SHA256,
                weights="sharded"),
    # The 12B replication model, BF16, one model.safetensors (no index). Its chat_template.jinja (36e3a42e...) and
    # tokenizer.json (cc8d3a0c...) are byte-identical to the 31B's; the pinned manifest binds them. The manifest
    # sha256 is the coordinator's record of 2026-11-01 (NEXT_ACTION.md).
    ServedModel(name="gemma4-12b", hub_id="google/gemma-4-12B-it", revision="0e2b1058541244490925fbacf8972041435691ac",
                alias="absent-edges-gemma4-12b-bf16", model_dir=str(PUBLIC_ROOT / "models/gemma4-12b-bf16-0e2b1058"),
                manifest=str(PUBLIC_ROOT / "models/COPY-gemma4-12b-bf16-0e2b1058.sha256"),
                manifest_sha256="06656cda567f7acc9a35a090632291e14c770ea5d989292910ebe15fb26621d4",
                weights="single"),
    # The main-track QAT checkpoint (DECISIONS.md 2026-11-02, an additional checkpoint test): compressed-tensors W4A16
    # (pack-quantized) in one model.safetensors. vLLM detects the quantization from config.json, so the flags stay the
    # frozen ones (--dtype bfloat16 is the activation dtype). Its own chat_template.jinja is newer and refuses
    # string-form tool-call arguments, so it is served with the frozen 31B BF16 template (with parsed arguments both
    # templates rendered identical prompts on 300/300 sampled requests): its prompts are those of the BF16 run. The
    # manifest sha256 is the coordinator's record of 2026-11-02.
    ServedModel(name="gemma4-31b-qat", hub_id="google/gemma-4-31B-it-qat-w4a16-ct",
                revision="52f3f65bc7a02d555763bc923bd1d9094898219d", alias="absent-edges-gemma4-31b-qat-w4a16",
                model_dir=str(PUBLIC_ROOT / "models/gemma4-31b-qat-w4a16-52f3f65b"),
                manifest=str(PUBLIC_ROOT / "models/COPY-gemma4-31b-qat-w4a16-52f3f65b.sha256"),
                manifest_sha256="067a141c41ecb627c6975e7effb8ca07abb9a2f24b28d14ed367d76d4f4f6cba",
                weights="single", chat_template=str(MODEL_CHAT_TEMPLATE),
                chat_template_sha256=MODEL_CHAT_TEMPLATE_SHA256),
    # Follow-up A8 (protocols/EXPA_A8_MODELS.md): three other model families, Apache-2.0, ungated, pinned at the Hub
    # revision below, downloaded to the coordinator's folder on Core2 and deleted after the runs. Each manifest lists
    # every top-level file of the copy with its sha256 (made at download, 2026-11-10).
    ServedModel(name="qwen3-coder-30b-a3b", hub_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                revision="b2cff646eb4bb1d68355c01b18ae02e7cf42d120", alias="absent-edges-qwen3-coder-30b-a3b",
                model_dir=f"{A8_MODELS}/qwen3-coder-30b-a3b-b2cff646",
                manifest=f"{A8_MODELS}/COPY-qwen3-coder-30b-a3b-b2cff646.sha256",
                manifest_sha256="b18d81069b996e07ce8f0358c10aeaad3b39074abcb3fef4da3c63e4d5fffd36",
                weights="sharded", tool_call_parser="qwen3_coder", reasoning_parser=None),
    # gpt-oss-20b (openai/gpt-oss-20b @ 6cee5e81) was dropped at its dev check: vLLM renders its prompts through the
    # harmony format, so the harness's /tokenize preflight never matches the served prompt (483 vs 495 tokens); the
    # accounting check is not relaxed. IBM's Granite 3.3 8B replaces it (protocols/EXPA_A8_MODELS.md, amendment 1).
    ServedModel(name="granite-3-3-8b", hub_id="ibm-granite/granite-3.3-8b-instruct",
                revision="51dd4bc2ade4059a6bd87649d68aa11e4fb2529b", alias="absent-edges-granite-3.3-8b",
                model_dir=f"{A8_MODELS}/granite-3.3-8b-instruct-51dd4bc2",
                manifest=f"{A8_MODELS}/COPY-granite-3.3-8b-instruct-51dd4bc2.sha256",
                manifest_sha256="68bd0b44cf899b756beef8136817224634eb1bb4650978407aab7e5f6b894a93",
                weights="sharded", tool_call_parser="granite", reasoning_parser=None),
    ServedModel(name="olmo-3-1-32b", hub_id="allenai/Olmo-3.1-32B-Instruct",
                revision="ac0587e4a7744a551c059d8cd17ba220bc940dae", alias="absent-edges-olmo-3.1-32b",
                model_dir=f"{A8_MODELS}/olmo-3.1-32b-instruct-ac0587e4",
                manifest=f"{A8_MODELS}/COPY-olmo-3.1-32b-instruct-ac0587e4.sha256",
                manifest_sha256="7e8ed5b873ae3dcbf7b9a0a0a2144bebd953f82a05ec578b79000aadc74b831c",
                weights="sharded", tool_call_parser="olmo3", reasoning_parser=None,
                # transformers 5.14 parses this config's rope settings per layer type, which vLLM 0.25.1 does not
                # read; the override restates the config's own values (YaRN on full attention, theta 500000).
                hf_overrides=OLMO3_ROPE_OVERRIDE),
    # Amendment 2 (protocols/EXPA_A8_MODELS.md): two more families after OLMo and Granite failed their dev checks.
    ServedModel(name="ernie-4-5-21b-a3b", hub_id="baidu/ERNIE-4.5-21B-A3B-PT",
                revision="87db95487941cb39592ee0abca3b9155a6d19c5c", alias="absent-edges-ernie-4.5-21b-a3b",
                model_dir=f"{A8_MODELS}/ernie-4.5-21b-a3b-87db9548",
                manifest=f"{A8_MODELS}/COPY-ernie-4.5-21b-a3b-87db9548.sha256",
                manifest_sha256="78f543528bb112ad6ee2b812c4716e43a6d25a436557b255f5705743496fb6ee",
                weights="sharded", tool_call_parser="ernie45", reasoning_parser=None),
    ServedModel(name="seed-oss-36b", hub_id="ByteDance-Seed/Seed-OSS-36B-Instruct",
                revision="497f1dca95ebdec98e41d517b9f060ee753c902f", alias="absent-edges-seed-oss-36b",
                model_dir=f"{A8_MODELS}/seed-oss-36b-497f1dca",
                manifest=f"{A8_MODELS}/COPY-seed-oss-36b-497f1dca.sha256",
                manifest_sha256="868bf345124a1b6aaf929169592e37b91a4371ac6b888e60d471e72fa56cbd77",
                weights="sharded", tool_call_parser="seed_oss", reasoning_parser="seed_oss"),
    # Amendment 3: two more families after ERNIE and Seed-OSS failed their dev checks.
    ServedModel(name="glm-4-5-air-fp8", hub_id="zai-org/GLM-4.5-Air-FP8",
                revision="f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc", alias="absent-edges-glm-4.5-air-fp8",
                model_dir=f"{A8_MODELS}/glm-4.5-air-fp8-f9a9c5ac",
                manifest=f"{A8_MODELS}/COPY-glm-4.5-air-fp8-f9a9c5ac.sha256",
                manifest_sha256="75a8b653c2098a96ce661b2ba390a5342b83916a5ba2234ab211c454cbb71f58",
                weights="sharded", tool_call_parser="glm45", reasoning_parser="glm45",
                extra_args=("--tensor-parallel-size", "2")),
    ServedModel(name="mistral-small-3-2-24b", hub_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
                revision="95a6d26c4bfb886c58daf9d3f7332c857cb27b43", alias="absent-edges-mistral-small-3.2-24b",
                model_dir=f"{A8_MODELS}/mistral-small-3.2-24b-95a6d26c",
                manifest=f"{A8_MODELS}/COPY-mistral-small-3.2-24b-95a6d26c.sha256",
                manifest_sha256="3a24ba5ed275c280d6cdb7923080c71a64bbea96f722f02a5054502b5847d7ce",
                weights="sharded", tool_call_parser="mistral", reasoning_parser=None,
                extra_args=("--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral")),
    # Amendment 5: the competition's two small Gemma 4 models (BF16) and two more families.
    ServedModel(name="gemma4-e4b", hub_id="google/gemma-4-E4B-it",
                revision="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c", alias="absent-edges-gemma4-e4b",
                model_dir=f"{A8_MODELS}/gemma4-e4b-4ecc2ba6", manifest=f"{A8_MODELS}/COPY-gemma4-e4b-4ecc2ba6.sha256",
                manifest_sha256="ec475ba70f020a2034476b1d6ba9af01ca05764f090ea54ec94c3174a42aa654", weights="single"),
    ServedModel(name="gemma4-e2b", hub_id="google/gemma-4-E2B-it",
                revision="4adf15748de9e1a0097b6274cee9c15836c41c37", alias="absent-edges-gemma4-e2b",
                model_dir=f"{A8_MODELS}/gemma4-e2b-4adf1574", manifest=f"{A8_MODELS}/COPY-gemma4-e2b-4adf1574.sha256",
                manifest_sha256="69014c7d0901450b6aed935844da68ea5df27e301130d2c7627e75d5db7edf1b", weights="single"),
    ServedModel(name="apertus-8b", hub_id="swiss-ai/Apertus-8B-Instruct-2509",
                revision="b946d40447b2b597999b9c86d44bee0b452c919f", alias="absent-edges-apertus-8b",
                model_dir=f"{A8_MODELS}/apertus-8b-b946d404", manifest=f"{A8_MODELS}/COPY-apertus-8b-b946d404.sha256",
                manifest_sha256="2301e28830864906b0521d037131be9049fc122bbfc562329b7bf0fb7f071108", weights="sharded",
                tool_call_parser="apertus", reasoning_parser=None),
    ServedModel(name="phi4-mini", hub_id="microsoft/Phi-4-mini-instruct",
                revision="cfbefacb99257ffa30c83adab238a50856ac3083", alias="absent-edges-phi4-mini",
                model_dir=f"{A8_MODELS}/phi4-mini-cfbefacb", manifest=f"{A8_MODELS}/COPY-phi4-mini-cfbefacb.sha256",
                manifest_sha256="ebf8771247cc1422bbeae3b59cb20a6255915dc956a75528d0a813902ae270b0", weights="sharded",
                tool_call_parser="phi4_mini_json", reasoning_parser=None),
    # Amendment 6: Meta's Muse Glimmer 30B (user request). vLLM 0.25.1 cannot serve its architecture, so this one
    # profile is served from a separate vLLM 0.33.0 environment (/mnt/kldata/arjun/absent-edges-a8/envs/vllm033).
    ServedModel(name="muse-glimmer-30b", hub_id="meta-models/Muse-Glimmer-30B",
                revision="cec94fcfd6512f2cfe708138c32533b346ea4f69", alias="absent-edges-muse-glimmer-30b",
                model_dir=f"{A8_MODELS}/muse-glimmer-30b-cec94fcf",
                manifest=f"{A8_MODELS}/COPY-muse-glimmer-30b-cec94fcf.sha256",
                manifest_sha256="404053b14603f4debdbd4f6a23eefd5ae6eb9d3f10420fbd53bd9106e0f43e05", weights="sharded",
                tool_call_parser="muse_glimmer", reasoning_parser="muse_glimmer"),
    # Amendment 7 (web search for further families with native tool calling; served from the vLLM 0.33.0 env).
    ServedModel(name="ministral3-14b", hub_id="mistralai/Ministral-3-14B-Instruct-2512",
                revision="29439f81c2be264d8d393273f99e7db9c0961120", alias="absent-edges-ministral3-14b",
                model_dir=f"{A8_MODELS}/ministral3-14b-29439f81", manifest=f"{A8_MODELS}/COPY-ministral3-14b-29439f81.sha256",
                manifest_sha256="4f53f4305670de4ca615e6fe311f6c51b0e0af58f8eaed13ae099a392565885a", weights="sharded",
                tool_call_parser="mistral", reasoning_parser=None),
    ServedModel(name="devstral2-24b", hub_id="mistralai/Devstral-Small-2-24B-Instruct-2512",
                revision="55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128", alias="absent-edges-devstral2-24b",
                model_dir=f"{A8_MODELS}/devstral2-24b-55c5b41e", manifest=f"{A8_MODELS}/COPY-devstral2-24b-55c5b41e.sha256",
                manifest_sha256="4fd243e66741442d8204ada02e4bc04418bcf4acf4710ac3d55a9971c358d8dc", weights="sharded",
                tool_call_parser="mistral", reasoning_parser=None),
    ServedModel(name="nemotron3-nano-30b", hub_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                revision="bf77c3174f68ad409e1c2aa60daeb46e32d1c606", alias="absent-edges-nemotron3-nano-30b",
                model_dir=f"{A8_MODELS}/nemotron3-nano-30b-bf77c317",
                manifest=f"{A8_MODELS}/COPY-nemotron3-nano-30b-bf77c317.sha256",
                manifest_sha256="2f7a4f164aee65026c55014003f7601e278750d791ee8d97cfd5edeb4a72ed71", weights="sharded",
                tool_call_parser="qwen3_coder", reasoning_parser="nemotron_v3"))}
# The A8 profiles (protocols/EXPA_A8_MODELS.md); every other profile is a Gemma 4 checkpoint.
A8_PROFILES = ("qwen3-coder-30b-a3b", "olmo-3-1-32b", "granite-3-3-8b", "ernie-4-5-21b-a3b", "seed-oss-36b",
               "glm-4-5-air-fp8", "mistral-small-3-2-24b", "gemma4-e4b", "gemma4-e2b", "apertus-8b", "phi4-mini",
               "muse-glimmer-30b", "ministral3-14b", "devstral2-24b", "nemotron3-nano-30b")


def served_model(name):
    """The model profile called ``name``."""
    if not isinstance(name, str) or name not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile {name!r}; known: {', '.join(sorted(MODEL_PROFILES))}")
    return MODEL_PROFILES[name]


@dataclass(frozen=True)
class ServeSpec:
    run: str
    instance: str
    gpu_uuid: str
    port: int
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    # Covers the whole D11.3 window (P by 2026-11-05 12:00 UTC, secondary blocks by 2026-11-06 12:00 UTC) from
    # an instance started on 2026-11-01; the runner unit allows 5 days per launch.
    runtime_max_seconds: int = SERVE_RUNTIME_MAX_SECONDS
    memory_max: str = "192G"
    startup_timeout_seconds: int = 1800
    watchdog_seconds: int = WATCHDOG_SECONDS
    reobserve_seconds: int = REOBSERVE_SECONDS
    serve_env: str = str(SERVE_ENV)
    model_dir: str = str(MODEL_DIR)
    model_manifest: str = str(MODEL_MANIFEST)
    model_manifest_sha256: str = MODEL_MANIFEST_SHA256
    model_check: str = "full"
    expected_user: str = EXPECTED_USER
    runtime_root: str = str(RUNTIME_ROOT)
    # None: record who owns the mirror (preflight "mirror_ownership"); an integer: refuse unless every entry of
    # the serve env, runtime and model trees and every ancestor is owned by that UID or root and not writable
    # by group or others (review finding: Core1 UID 1006, another person's account, owns the mirror and
    # /mnt/kldata over NFS). Set 0 once the coordinator has made the chain root-owned.
    mirror_owner_uid: int = None
    # The served model (MODEL_PROFILES): alias, revision and required files. The model_* paths above default to the
    # default profile's copy; ServeSpec.for_profile fills another profile's own copy.
    model_profile: str = DEFAULT_MODEL_PROFILE

    @classmethod
    def for_profile(cls, model_profile=DEFAULT_MODEL_PROFILE, **values):
        """A spec for ``model_profile`` whose model_dir, model_manifest and model_manifest_sha256 default (when absent
        or None) to that profile's pinned public-mirror copy; every path stays a parameter."""
        served = served_model(model_profile)
        for key, default in (("model_dir", served.model_dir), ("model_manifest", served.manifest),
                             ("model_manifest_sha256", served.manifest_sha256)):
            if values.get(key) is None:
                values[key] = default
        return cls(model_profile=model_profile, **values)

    @property
    def served_model(self):
        return served_model(self.model_profile)

    def __post_init__(self):
        served = served_model(self.model_profile)
        # A spec never names another profile's model files: the 12B profile with the 31B copy would serve the 31B
        # under the 12B alias and pass the 31B manifest pin.
        for other in MODEL_PROFILES.values():
            if other.name != served.name and (self.model_dir == other.model_dir or self.model_manifest == other.manifest
                                              or self.model_manifest_sha256 == other.manifest_sha256):
                raise ValueError(f"model profile {served.name} names the model files of profile {other.name}; "
                                 "use ServeSpec.for_profile")
        for name in ("run", "instance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _NAME.fullmatch(value):
                raise ValueError(f"{name} must match [a-z0-9][a-z0-9-]{{0,47}}")
        if not isinstance(self.gpu_uuid, str) or not _UUID.fullmatch(self.gpu_uuid):
            raise ValueError("gpu_uuid must be a full lowercase GPU UUID (never an index)")
        if self.gpu_uuid in FORBIDDEN_GPU_UUIDS:
            raise ServePreflightError("GPU5 is never used")
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise ValueError("port must be an integer between 1024 and 65535")
        if type(self.max_num_seqs) is not int or not 1 <= self.max_num_seqs <= 256:
            raise ValueError("max_num_seqs must be between 1 and 256")
        if not isinstance(self.gpu_memory_utilization, float) or not 0.1 <= self.gpu_memory_utilization <= 0.95:
            raise ValueError("gpu_memory_utilization must be a float between 0.1 and 0.95")
        for name in ("runtime_max_seconds", "startup_timeout_seconds", "watchdog_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.reobserve_seconds) is not int or not 0 <= self.reobserve_seconds <= 60:
            raise ValueError("reobserve_seconds must be an integer between 0 and 60")
        if not isinstance(self.memory_max, str) or not re.fullmatch(r"[0-9]+[KMGT]?|infinity", self.memory_max):
            raise ValueError("memory_max must look like 192G (or infinity)")
        for name in ("serve_env", "model_dir", "model_manifest", "runtime_root"):
            _absolute(name, getattr(self, name))
        if self.mirror_owner_uid is not None and (type(self.mirror_owner_uid) is not int or self.mirror_owner_uid < 0):
            raise ValueError("mirror_owner_uid must be a non-negative integer or None")
        if self.model_manifest_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", str(self.model_manifest_sha256)):
            raise ValueError("model_manifest_sha256 must be 64 lowercase hex digits (or None in tests)")
        if self.model_check not in ("full", "quick"):
            raise ValueError("model_check must be full or quick")
        if not isinstance(self.expected_user, str) or not re.fullmatch(r"[a-z_][a-z0-9_.-]{0,31}", self.expected_user):
            raise ValueError("expected_user must be a user name")

    @property
    def run_root(self):
        return CACHE_PARENT / f"ae-expA-{self.run}"

    @property
    def cache_root(self):
        return self.run_root / "serve" / self.instance

    @property
    def unit(self):
        return f"ae-expA-serve-{self.run}-{self.instance}"

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.port}"

    @property
    def serve_python(self):
        return Path(self.serve_env) / "bin" / "python"


# ---------------------------------------------------------------------------
# Identity guard (finding F3)
# ---------------------------------------------------------------------------

def _host():
    return socket.gethostname().split(".")[0]


def current_identity():
    euid = os.geteuid()
    try:
        user = pwd.getpwuid(euid).pw_name
    except KeyError:
        user = None
    return {"uid": os.getuid(), "euid": euid, "gid": os.getgid(), "egid": os.getegid(), "user": user,
            "host": _host()}


def identity_failures(identity, expected_user=EXPECTED_USER):
    """Empty only when the effective user is ``expected_user`` with its pinned UID on this host.

    Root is refused, and Core1 UID 1006 (another person's account) is refused by
    number whatever name it resolves to.
    """
    failures = []
    host = (identity.get("host") or "").split(".")[0]
    euid = identity.get("euid")
    if euid == 0 or identity.get("uid") == 0:
        failures.append("root_forbidden")
    if identity.get("uid") != euid:
        failures.append("uid_euid_mismatch")
    if euid in FORBIDDEN_UIDS_BY_HOST.get(host, ()):
        failures.append(f"uid_{euid}_forbidden_on_{host}")
    if identity.get("user") != expected_user:
        failures.append(f"effective_user_{identity.get('user')}_is_not_{expected_user}")
    pinned = EXPECTED_UID_BY_HOST.get(host) if expected_user == EXPECTED_USER else None
    if pinned is not None and euid != pinned:
        failures.append(f"uid_{euid}_is_not_{expected_user}_uid_{pinned}_on_{host}")
    return failures


def require_identity(expected_user=EXPECTED_USER, identity=None):
    identity = current_identity() if identity is None else identity
    failures = identity_failures(identity, expected_user)
    if failures:
        raise ServePreflightError("refusing to run serving as this identity: " + ", ".join(failures))
    return identity


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def vllm_argv(spec: ServeSpec):
    """The frozen flags for every profile; a profile's chat-template override adds ``--chat-template PATH`` last."""
    argv = [str(spec.serve_python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", spec.model_dir, "--served-model-name", spec.served_model.alias, "--dtype", "bfloat16",
            "--host", "127.0.0.1", "--port", str(spec.port), "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", repr(spec.gpu_memory_utilization), "--max-num-seqs", str(spec.max_num_seqs),
            "--language-model-only", "--enable-auto-tool-choice", "--tool-call-parser",
            spec.served_model.tool_call_parser]
    if spec.served_model.reasoning_parser is not None:
        argv += ["--reasoning-parser", spec.served_model.reasoning_parser]
    argv += ["--generation-config", "vllm"]
    if spec.served_model.prefix_caching:
        argv += ["--enable-prefix-caching"]
    if spec.served_model.hf_overrides is not None:
        argv += ["--hf-overrides", spec.served_model.hf_overrides]
    argv += list(spec.served_model.extra_args)
    if spec.served_model.chat_template is not None:
        argv += ["--chat-template", spec.served_model.chat_template]
    return argv


def serve_environment(spec: ServeSpec):
    """Allowlisted environment (no inherited secrets); TMPDIR, HOME and every cache under the run dir.

    PATH starts with the serve env's bin/, which provides ninja (smoke run a failed without it).
    """
    root = spec.cache_root
    cache = root / "cache"
    env = {"PATH": f"{spec.serve_env}/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(root / "home"),
           "TMPDIR": str(root / "tmp"), "TMP": str(root / "tmp"), "TEMP": str(root / "tmp"),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1",
           "DO_NOT_TRACK": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": spec.gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    for name in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "HF_HOME", "HF_HUB_CACHE",
                 "PIP_CACHE_DIR", "UV_CACHE_DIR", "PYTHONPYCACHEPREFIX", "CUDA_CACHE_PATH",
                 "FLASHINFER_WORKSPACE_BASE", "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "TRITON_CACHE_DIR",
                 "TORCHINDUCTOR_CACHE_DIR"):
        env[name] = str(cache / name.lower())
    return env


def unit_directories(spec: ServeSpec):
    """Directories the supervisor creates (0700) inside the unit's only writable path."""
    root = spec.cache_root
    directories = [root, root / "logs", root / "records", root / "tmp", root / "home", root / "cache"]
    for value in serve_environment(spec).values():
        if value.startswith(str(root) + "/") and Path(value) not in directories:
            directories.append(Path(value))
    return directories


def hardening_properties(read_write_paths, *, read_only_paths=(), extra=()):
    """The unit sandbox verified by the serving smoke, plus explicit extra properties.

    ProtectSystem=strict leaves the API file systems (/dev, so /dev/shm) writable; ReadOnlyPaths=/dev/shm
    makes every /dev/shm entry read-only, and the more specific ReadWritePaths= reopen only the unit's own
    directories (review finding on /dev/shm).
    """
    properties = ["NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes",
                  "InaccessiblePaths=" + " ".join("-" + path for path in INACCESSIBLE_PATHS), "UMask=0077",
                  f"ReadOnlyPaths={CACHE_PARENT}"]
    properties += ["ReadOnlyPaths=" + str(path) for path in read_only_paths]
    properties += ["ReadWritePaths=" + str(path) for path in read_write_paths]
    return properties + list(extra)


# vLLM creates POSIX semaphores and shared-memory segments directly in /dev/shm (multiprocessing SemLock,
# shm_broadcast), which ReadOnlyPaths=/dev/shm refuses with EROFS (pilot-20261101a, all four instances).
# The serve unit therefore gets a private tmpfs at /dev/shm, and only the staged release (read-only) and
# its own instance directory (read-write) of the run root are bound into it: the rest of the host's
# /dev/shm, including the runner's inputs and other runs, is not even visible to the unit.
SERVE_SHM_TMPFS = f"TemporaryFileSystem={CACHE_PARENT}:rw,nosuid,nodev,mode=1777,size=64G"


def serve_hardening_properties(release_root, cache_root):
    """The serve unit's sandbox: hardening_properties without the shared /dev/shm, plus a private one."""
    properties = [value for value in hardening_properties([]) if value != f"ReadOnlyPaths={CACHE_PARENT}"]
    return properties + [SERVE_SHM_TMPFS, f"BindReadOnlyPaths={release_root}", f"BindPaths={cache_root}"]


def spec_json(spec: ServeSpec):
    """The unit's ``supervise --spec``. The default model profile is left out, so a default plan starts the
    byte-identical unit command of the frozen release (a53c49396); ServeSpec fills the default back in."""
    value = asdict(spec)
    if value["model_profile"] == DEFAULT_MODEL_PROFILE:
        del value["model_profile"]
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def systemd_run_argv(spec: ServeSpec, *, release_root, runtime_python=str(RUNTIME_PYTHON)):
    """The transient system unit (run with sudo). Printed for the coordinator; never executed here."""
    release_root = Path(_absolute("release_root", str(release_root)))
    runtime_python = _absolute("runtime_python", str(runtime_python))
    root = spec.cache_root
    # --collect: a failed unit (vLLM crash, preflight refusal, collision) is unloaded, so `serve` can start the
    # instance again under the same name without a manual `systemctl reset-failed`.
    argv = ["systemd-run", f"--unit={spec.unit}", "--collect",
            f"--description=Absent Edges Exp A vLLM {spec.run}/{spec.instance} ({spec.gpu_uuid})"]
    properties = ["Type=exec", f"User={spec.expected_user}", f"Group={spec.expected_user}",
                  f"WorkingDirectory={root}", "KillMode=control-group", "KillSignal=SIGTERM", "Restart=no",
                  "TimeoutStopSec=180", f"RuntimeMaxSec={spec.runtime_max_seconds}",
                  f"MemoryMax={spec.memory_max}", "MemorySwapMax=0"]
    properties += serve_hardening_properties(release_root, root)
    argv += ["--property=" + value for value in properties]
    argv += [f"--setenv=PATH={spec.serve_env}/bin:/usr/bin:/bin", "--setenv=LANG=C.UTF-8",
             "--setenv=PYTHONDONTWRITEBYTECODE=1", f"--setenv=HOME={root / 'home'}", f"--setenv=TMPDIR={root / 'tmp'}",
             "--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID", "--"]
    argv += [runtime_python, "-I", "-B", str(release_root / "scripts" / "expA_serve.py"), "supervise",
             "--spec", spec_json(spec)]
    return argv


def plan(spec: ServeSpec, *, release_root=None, runtime_python=str(RUNTIME_PYTHON), identity=None):
    """Everything the operator runs for one instance. Refuses unless run as thilak.arjun."""
    who = require_identity(spec.expected_user, identity)
    release_root = release_root or str(spec.run_root / "inputs" / "release")
    command = systemd_run_argv(spec, release_root=release_root, runtime_python=runtime_python)
    prepare = ["install", "-d", "-m", "0700", str(spec.run_root), str(spec.run_root / "serve"), str(spec.cache_root)]
    return {"protocol": SERVE_PROTOCOL, "spec": asdict(spec), "unit": spec.unit, "endpoint": spec.endpoint,
            "run_root": str(spec.run_root), "cache_root": str(spec.cache_root), "planned_by": who,
            "prepare_argv": prepare, "prepare_command": shlex.join(prepare),
            "vllm_argv": vllm_argv(spec), "environment": serve_environment(spec),
            "systemd_run_argv": command, "systemd_run_command": "sudo " + shlex.join(command),
            "flags_source": SOURCE_TEMPLATE, "smoke_reference": SMOKE_REFERENCE,
            "model": {"path": spec.model_dir, "alias": spec.served_model.alias, "revision": spec.served_model.revision,
                      "max_model_len": MAX_MODEL_LEN, "manifest": spec.model_manifest,
                      "manifest_sha256": spec.model_manifest_sha256, "check": spec.model_check,
                      **spec.served_model.describe()},
            "preflight_rules": {"expected_user": spec.expected_user,
                                "expected_uid_by_host": dict(EXPECTED_UID_BY_HOST),
                                "forbidden_uids_by_host": {host: sorted(uids) for host, uids
                                                           in FORBIDDEN_UIDS_BY_HOST.items()},
                                "forbidden_uuids": sorted(FORBIDDEN_GPU_UUIDS),
                                "forbidden_indices": sorted(FORBIDDEN_GPU_INDICES),
                                "forbidden_minor_numbers": sorted(FORBIDDEN_GPU_MINORS),
                                "max_idle_used_mib": MAX_IDLE_USED_MIB, "processes_allowed": 0,
                                "cache_filesystem": "tmpfs"},
            "watchdog": {"interval_seconds": spec.watchdog_seconds, "reobserve_seconds": spec.reobserve_seconds,
                         "rule": "a GPU PID not provably ours (descendant or our process group) on two consecutive "
                                 "observations is foreign; hidden /proc entries are never assumed ours",
                         "action": "record a collision, stop only our own process group, exit (ends only this unit)"},
            "notes": ["Create the unit's directory as thilak.arjun first (prepare_command): ReadWritePaths must exist.",
                      "One instance per GPU and one port per instance; the supervisor refuses a busy GPU or port.",
                      "Supervisor records: <cache_root>/records/*.json; journal: journalctl -u " + spec.unit]}


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------

def parse_sha256sums(text):
    """{name: sha256} from `sha256sum` output for files in one directory (no paths)."""
    entries = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = _SUMS_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"manifest line {number} is not sha256sum output")
        name = match.group(2)
        if "/" in name or name in (".", "..") or name.startswith("-"):
            raise ValueError(f"manifest line {number} names {name!r}, not a file in the model directory")
        if name in entries:
            raise ValueError(f"manifest lists {name!r} twice")
        entries[name] = match.group(1)
    if not entries:
        raise ValueError("manifest is empty")
    return entries


def sha256_file(path, block=8 * 1024**2):
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        while True:
            chunk = os.read(descriptor, block)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_edges(path, size, span=1024**2):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        head = os.read(descriptor, span)
        os.lseek(descriptor, max(0, size - span), os.SEEK_SET)
        tail = os.read(descriptor, span)
    finally:
        os.close(descriptor)
    if size and (not head or not tail):
        raise OSError("short read")


def verify_model(model_dir, manifest, *, mode="full", expected_manifest_sha256=MODEL_MANIFEST_SHA256,
                 workers=4, quick_hash_limit=QUICK_HASH_LIMIT, required_files=MODEL_REQUIRED_FILES):
    """Check every file of the manifest is a readable regular file whose sha256 matches.

    ``full`` hashes every byte (about 62 GB for the Gemma copy). ``quick`` hashes
    files up to ``quick_hash_limit`` bytes and only proves the larger ones readable
    (first and last MiB); it is recorded as such and is never the supervisor's
    default. ``required_files`` must all be listed (a profile's required_files;
    the default is the sharded 31B copy's).
    """
    if mode not in ("full", "quick"):
        raise ValueError("mode must be full or quick")
    model_dir, manifest = Path(model_dir), Path(manifest)
    failures = []
    record = {"model_dir": str(model_dir), "manifest": str(manifest), "mode": mode, "manifest_sha256": None,
              "expected_manifest_sha256": expected_manifest_sha256, "files": {}, "unexpected_files": [],
              "required_files": list(required_files), "failures": failures}
    started = time.monotonic()
    try:
        raw = manifest.read_bytes()
    except OSError as error:
        failures.append(f"model_manifest_unreadable:{error.strerror or type(error).__name__}")
        return record
    record["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and record["manifest_sha256"] != expected_manifest_sha256:
        failures.append("model_manifest_sha256_mismatch")
        return record
    try:
        entries = parse_sha256sums(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        failures.append("model_manifest_invalid:" + str(error)[:200])
        return record
    for name in required_files:
        if name not in entries:
            failures.append("model_manifest_lacks:" + name)
    try:
        listing = sorted(os.listdir(model_dir))
    except OSError as error:
        failures.append(f"model_dir_unreadable:{error.strerror or type(error).__name__}")
        return record
    record["unexpected_files"] = sorted(set(listing) - set(entries) - MODEL_ALLOWED_EXTRA)
    for name in record["unexpected_files"]:
        failures.append("model_unexpected_file:" + name)
    jobs = []
    for name in sorted(entries):
        path = model_dir / name
        entry = {"expected_sha256": entries[name], "bytes": None, "sha256": None, "verified": False}
        record["files"][name] = entry
        try:
            info = os.lstat(path)
        except OSError as error:
            failures.append(f"model_file_missing:{name}:{error.strerror or type(error).__name__}")
            continue
        if not stat.S_ISREG(info.st_mode):
            failures.append("model_file_not_regular:" + name)
            continue
        entry["bytes"] = info.st_size
        jobs.append((name, path, info.st_size))

    def check(job):
        name, path, size = job
        try:
            if mode == "full" or size <= quick_hash_limit:
                return name, sha256_file(path), None
            _read_edges(path, size)
            return name, None, None
        except OSError as error:
            return name, None, f"model_file_unreadable:{name}:{error.strerror or type(error).__name__}"

    jobs.sort(key=lambda job: -job[2])
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for name, digest, error in pool.map(check, jobs):
            entry = record["files"][name]
            if error:
                failures.append(error)
                continue
            entry["sha256"] = digest
            if digest is None:
                entry["readable_only"] = True
            elif digest == entry["expected_sha256"]:
                entry["verified"] = True
            else:
                failures.append("model_file_sha256_mismatch:" + name)
    record["seconds"] = round(time.monotonic() - started, 3)
    record["sha256_verified_files"] = sum(1 for entry in record["files"].values() if entry["verified"])
    return record


def verify_chat_template(path, expected_sha256):
    """Check a profile's chat-template override: a regular file (never a symlink) whose sha256 is the pinned one.

    The record's ``failures`` is empty only then: ``chat_template_missing`` (absent or not reachable),
    ``chat_template_not_regular``, ``chat_template_unreadable`` or ``chat_template_sha256_mismatch``.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("a chat-template override pins its sha256 (64 lowercase hex digits)")
    failures = []
    record = {"path": str(path), "expected_sha256": expected_sha256, "sha256": None, "bytes": None,
              "failures": failures}
    try:
        info = os.lstat(path)
    except OSError as error:
        failures.append(f"chat_template_missing:{error.strerror or type(error).__name__}")
        return record
    if not stat.S_ISREG(info.st_mode):
        failures.append("chat_template_not_regular")
        return record
    try:
        record["sha256"] = sha256_file(path)
    except OSError as error:
        failures.append(f"chat_template_unreadable:{error.strerror or type(error).__name__}")
        return record
    record["bytes"] = info.st_size
    if record["sha256"] != expected_sha256:
        failures.append("chat_template_sha256_mismatch")
    return record


# ---------------------------------------------------------------------------
# GPU observation (nvidia-smi, read-only)
# ---------------------------------------------------------------------------

def _mib(text):
    match = _MIB.fullmatch(text or "")
    if match is None:
        raise ValueError(f"unexpected memory value {text!r}")
    return int(match.group(1))


def parse_gpu_xml(xml_bytes, uuid):
    """{uuid, minor_number, used_mib, total_mib, pids} for ``uuid`` from `nvidia-smi -q -x`."""
    if not isinstance(xml_bytes, bytes) or len(xml_bytes) > 8 * 1024**2:
        raise ValueError("nvidia-smi XML must be bounded bytes")
    root = ET.fromstring(xml_bytes)
    if root.tag != "nvidia_smi_log":
        raise ValueError("not an nvidia-smi XML log")
    matches = [gpu for gpu in root.findall("gpu") if (gpu.findtext("uuid") or "").strip() == uuid]
    if len(matches) != 1:
        raise ValueError("selected GPU UUID not found exactly once")
    gpu = matches[0]
    minor = (gpu.findtext("minor_number") or "").strip()
    if not minor.isdigit():
        raise ValueError("GPU minor number missing")
    processes = gpu.find("processes")
    if processes is None:
        raise ValueError("GPU process inventory missing")
    pids = []
    for process in processes.findall("process_info"):
        text = (process.findtext("pid") or "").strip()
        if not text.isdigit():
            raise ValueError("unparseable GPU process pid")
        pids.append(int(text))
    return {"uuid": uuid, "minor_number": int(minor), "used_mib": _mib(gpu.findtext("fb_memory_usage/used")),
            "total_mib": _mib(gpu.findtext("fb_memory_usage/total")), "pids": sorted(set(pids))}


def parse_compute_apps(csv_text, uuid):
    """PIDs from `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader` for ``uuid``."""
    pids = set()
    for line in csv_text.splitlines():
        if not line.strip() or line.strip().lower().startswith("no running"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2 or not fields[1].isdigit():
            raise ValueError("unexpected compute-apps row")
        if fields[0] == uuid:
            pids.add(int(fields[1]))
    return sorted(pids)


def parse_gpu_table(csv_text):
    """[{index, uuid, used_mib}] from `nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits`."""
    rows = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit() or not _UUID.fullmatch(fields[1]):
            raise ValueError("unexpected query-gpu row")
        used = int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else None
        rows.append({"index": int(fields[0]), "uuid": fields[1], "used_mib": used})
    if len({row["uuid"] for row in rows}) != len(rows) or len({row["index"] for row in rows}) != len(rows):
        raise ValueError("duplicate GPU index or UUID")
    return rows


def parse_gpu_index(csv_text, uuid):
    """nvidia-smi index of ``uuid`` from `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`."""
    found = [row["index"] for row in parse_gpu_table(csv_text) if row["uuid"] == uuid]
    if len(found) != 1:
        raise ValueError("selected GPU UUID not found exactly once in the index table")
    return found[0]


def _query(run, argv, timeout):
    return run(argv, stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=timeout).stdout


def observe_gpu(uuid, run=subprocess.run, timeout=20):
    """Read-only nvidia-smi observation of one GPU: index, minor number, memory and every listed PID."""
    xml = _query(run, ["nvidia-smi", "-q", "-x", "-i", uuid], timeout)
    apps = _query(run, ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], timeout)
    index = _query(run, ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], timeout)
    observation = parse_gpu_xml(xml, uuid)
    observation["pids"] = sorted(set(observation["pids"]) | set(parse_compute_apps(apps.decode(), uuid)))
    observation["index"] = parse_gpu_index(index.decode(), uuid)
    observation["observed_unix"] = time.time()
    return observation


def gpu_table(run=subprocess.run, timeout=20):
    """Read-only recheck of every GPU: idle (<=128 MiB, no process) and eligible (idle and not GPU5)."""
    table = parse_gpu_table(_query(run, ["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                         "--format=csv,noheader,nounits"], timeout).decode())
    apps = _query(run, ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], timeout).decode()
    for row in table:
        row["pids"] = parse_compute_apps(apps, row["uuid"])
        row["gpu5"] = row["uuid"] in FORBIDDEN_GPU_UUIDS or row["index"] in FORBIDDEN_GPU_INDICES
        row["idle"] = row["used_mib"] is not None and row["used_mib"] <= MAX_IDLE_USED_MIB and not row["pids"]
        row["eligible"] = row["idle"] and not row["gpu5"]
    return {"observed_unix": time.time(), "host": _host(), "gpus": table,
            "eligible": [row["uuid"] for row in table if row["eligible"]]}


def forbidden_gpu(observation, uuid=None):
    """Reasons this observation names GPU5 (by UUID, index or minor number)."""
    reasons = []
    if uuid in FORBIDDEN_GPU_UUIDS or observation.get("uuid") in FORBIDDEN_GPU_UUIDS:
        reasons.append("gpu5_uuid_forbidden")
    if observation.get("index") in FORBIDDEN_GPU_INDICES:
        reasons.append("gpu_index_5_forbidden")
    if observation.get("minor_number") in FORBIDDEN_GPU_MINORS:
        reasons.append("gpu_minor_5_forbidden")
    return reasons


def check_idle(observation, uuid):
    """Pre-launch rule: not GPU5 (UUID, index, minor), index known, <=128 MiB used, no processes."""
    failures = forbidden_gpu(observation, uuid)
    if observation.get("uuid") != uuid:
        failures.append("uuid_mismatch")
    if observation.get("index") is None:
        failures.append("gpu_index_unknown")
    if observation["used_mib"] > MAX_IDLE_USED_MIB:
        failures.append(f"gpu_memory_in_use_{observation['used_mib']}MiB")
    if observation["pids"]:
        failures.append("gpu_has_processes")
    return {"passed": not failures, "failures": failures, "observation": observation}


def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def check_cache_location(spec, mounts=None, magic=None):
    location = require_local_filesystem(spec.cache_root, mounts=mounts, magic=magic)
    if location["fstype"] != "tmpfs":
        raise ServePreflightError(f"{spec.cache_root} must be on local tmpfs, not {location['fstype']}")
    return location


def proc_mount_options(run=subprocess.run):
    """`findmnt -no OPTIONS /proc`, recorded. hidepid never weakens the watchdog (it fails closed)."""
    try:
        output = run(["findmnt", "-no", "OPTIONS", "/proc"], stdin=subprocess.DEVNULL, capture_output=True,
                     check=True, timeout=20).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": "findmnt -no OPTIONS /proc", "options": None, "hidepid": None,
                "error": f"{type(error).__name__}: {error}"[:300], "watchdog": "fail_closed"}
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    hidepid = None
    for line in lines:
        for option in line.split(","):
            if option.startswith("hidepid="):
                hidepid = option.split("=", 1)[1]
    return {"command": "findmnt -no OPTIONS /proc", "options": lines, "hidepid": hidepid, "error": None,
            "watchdog": "fail_closed"}


def _owner_entry(path):
    try:
        info = os.lstat(path)
    except OSError as error:
        return {"path": str(path), "error": type(error).__name__}
    return {"path": str(path), "uid": info.st_uid, "gid": info.st_gid, "mode": f"{stat.S_IMODE(info.st_mode):o}",
            "kind": "dir" if stat.S_ISDIR(info.st_mode) else "symlink" if stat.S_ISLNK(info.st_mode) else "file",
            "group_or_other_writable": bool(info.st_mode & 0o022) and not stat.S_ISLNK(info.st_mode)}


def mirror_roots(spec):
    roots = [spec.serve_env, spec.runtime_root, spec.model_dir, spec.model_manifest]
    if spec.served_model.chat_template is not None:
        roots.append(spec.served_model.chat_template)     # the override is served too
    return roots


def mirror_ownership(spec, *, full=False, limit=20):
    """Who can change the code and weights this unit runs (review finding on the Core1 view of the mirror).

    Always: the owner, group and mode of each root (serve env, runtime Python, model dir and manifest, and a
    profile's chat-template override) and of every ancestor up to /. With ``full``: every entry below the roots as
    well (lstat only, no hashing). When ``spec.mirror_owner_uid`` is set, an entry fails unless it is owned by that
    UID or by root and is not group- or other-writable (symlinks are judged by their directory); a failing ancestor
    could rename the whole mirror away, so it fails too.
    """
    chain, seen = [], set()
    for root in mirror_roots(spec):
        current = Path(root)
        while True:
            if str(current) not in seen:
                seen.add(str(current))
                chain.append(_owner_entry(current))
            if current.parent == current:
                break
            current = current.parent
    allowed = None if spec.mirror_owner_uid is None else {0, spec.mirror_owner_uid}

    def bad(entry):
        return allowed is not None and ("error" in entry or entry["uid"] not in allowed
                                        or entry["group_or_other_writable"])
    failures = [entry for entry in chain if bad(entry)]
    entries = len(chain)
    if full:
        for root in mirror_roots(spec):
            if not Path(root).is_dir():
                continue
            for current, dirnames, filenames in os.walk(root, followlinks=False):
                for name in dirnames + filenames:
                    entries += 1
                    entry = _owner_entry(Path(current) / name)
                    if bad(entry) and len(failures) < 10 * limit:
                        failures.append(entry)
    owners = sorted({entry.get("uid") for entry in chain if "uid" in entry})
    return {"mirror_owner_uid": spec.mirror_owner_uid, "checked": "full" if full else "roots_and_ancestors",
            "entries": entries, "chain": chain, "owners_in_chain": owners, "failures": failures[:limit],
            "failure_count": len(failures), "passed": not failures}


def check_serve_env(spec):
    failures = []
    bin_dir = Path(spec.serve_env) / "bin"
    for name in ("python", "ninja"):
        path = bin_dir / name
        if not (path.is_file() and os.access(path, os.X_OK)):
            failures.append(f"serve_env_lacks_executable:{name}")
    return failures


def preflight(spec: ServeSpec, *, observe=observe_gpu, port_free=port_is_free, mounts=None, magic=None,
              identity=None, proc_options=proc_mount_options, verify=verify_model):
    """Cheap checks first; the model hash runs only when everything else passed.

    A profile's chat-template override is one of the cheap checks: it must be there with its pinned sha256
    (recorded as ``chat_template_override``, a key only profiles with an override add).
    """
    who = current_identity() if identity is None else identity
    failures = identity_failures(who, spec.expected_user)
    try:
        gpu = check_idle(observe(spec.gpu_uuid), spec.gpu_uuid)
        failures += gpu["failures"]
        observation = gpu["observation"]
    except _OBSERVATION_ERRORS as error:
        failures.append("gpu_observation_failed:" + f"{type(error).__name__}: {error}"[:200])
        observation = None
    if not port_free(spec.port):
        failures.append("port_in_use")
    try:
        cache = check_cache_location(spec, mounts=mounts, magic=magic)
    except (RuntimeError, OSError) as error:
        failures.append("cache_location:" + str(error)[:200])
        cache = None
    failures += check_serve_env(spec)
    served = spec.served_model
    template = None
    if served.chat_template is not None:
        template = verify_chat_template(served.chat_template, served.chat_template_sha256)
        failures += template["failures"]
    proc = proc_options()
    ownership = mirror_ownership(spec, full=spec.mirror_owner_uid is not None)
    if not ownership["passed"]:
        failures.append(f"mirror_not_owned_by_uid_{spec.mirror_owner_uid}_or_root:"
                        + ",".join(entry["path"] for entry in ownership["failures"][:3]))
    if failures:
        model = {"skipped": "earlier preflight failures"}
    else:
        model = verify(spec.model_dir, spec.model_manifest, mode=spec.model_check,
                       expected_manifest_sha256=spec.model_manifest_sha256,
                       required_files=served.required_files)
        failures += model["failures"]
    result = {"protocol": SERVE_PROTOCOL, "passed": not failures, "failures": failures, "identity": who,
              "gpu": observation, "cache": cache, "proc_mount": proc, "mirror_ownership": ownership, "model": model,
              "spec": asdict(spec), "checked_unix": time.time()}
    if template is not None:
        result["chat_template_override"] = template
    return result


# ---------------------------------------------------------------------------
# Watchdog (finding F4)
# ---------------------------------------------------------------------------

def _stat_fields(pid, proc=Path("/proc")):
    raw = (proc / str(pid) / "stat").read_text()
    return raw[raw.rfind(")") + 2:].split()


def _pgid_of(pid, proc=Path("/proc")):
    return int(_stat_fields(pid, proc)[2])


def descendants(root_pid, proc=Path("/proc")):
    """PIDs whose parent chain reaches ``root_pid`` (one /proc scan; unreadable entries are skipped)."""
    parents = {}
    for entry in proc.iterdir():
        if entry.name.isdigit():
            try:
                parents[int(entry.name)] = int(_stat_fields(entry.name, proc)[1])
            except (OSError, ValueError, IndexError):
                continue
    found, frontier = set(), {root_pid}
    while frontier:
        children = {pid for pid, parent in parents.items() if parent in frontier and pid not in found}
        found |= children
        frontier = children
    return found


def provably_ours(pid, own_pgid, own_pids=(), pgid_of=_pgid_of):
    """True only with positive evidence: a listed own PID or a readable process group equal to ours."""
    if pid in own_pids:
        return True
    try:
        return pgid_of(pid) == own_pgid
    except (OSError, ValueError, IndexError):
        return False  # hidden (hidepid), vanished or unreadable: never assumed ours


def unproven_pids(observation, own_pgid, own_pids=(), pgid_of=_pgid_of):
    """GPU PIDs that are not provably ours. Foreign once they persist across one re-observation."""
    return sorted(pid for pid in observation["pids"] if not provably_ours(pid, own_pgid, own_pids, pgid_of))


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class Supervisor:
    """Unit entry: launch vLLM in its own process group, then watch the GPU every 30 s."""

    def __init__(self, spec: ServeSpec, *, observe=observe_gpu, sleep=time.sleep, popen=subprocess.Popen,
                 http_get=None, pgid_of=_pgid_of, own_descendants=descendants, preflight_check=None):
        self.spec, self.observe, self.sleep, self.popen, self.pgid_of = spec, observe, sleep, popen, pgid_of
        self.own_descendants = own_descendants
        self.http_get = http_get or self._http_get
        self.preflight_check = preflight_check
        self.records = spec.cache_root / "records"
        self.stop_requested = False
        self.child = None
        self.child_pgid = None

    def _http_get(self, path, timeout=10):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(self.spec.endpoint + path, timeout=timeout) as response:
            body = response.read(16 * 1024**2)
        return json.loads(body) if body else None

    def record(self, name, value):
        return write_immutable_json(self.records / f"{_utc()}-{name}.json",
                                    {"protocol": SERVE_PROTOCOL, "run": self.spec.run, "instance": self.spec.instance,
                                     "unit": self.spec.unit, "gpu_uuid": self.spec.gpu_uuid,
                                     "recorded_unix": time.time(), **value})

    def stop_own_group(self, reason):
        """Terminate only our vLLM process group; escalate to SIGKILL after 120 s.

        The group is signalled only while its leader (our child) is unreaped, so its
        PGID can never have been reused by another process. Anything left after the
        leader has gone is ended by the unit's KillMode=control-group when we exit.
        """
        if self.child is None or self.child_pgid is None:
            return
        group = self.child_pgid
        if group == os.getpgrp() or group <= 1:
            raise RuntimeError("refusing to signal the supervisor's own process group")
        if self.child.poll() is None:
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.child.wait()
        self.record("stopped", {"reason": reason, "pgid": group, "returncode": self.child.returncode})

    def _observe_twice_on_failure(self):
        try:
            return self.observe(self.spec.gpu_uuid), None
        except _OBSERVATION_ERRORS as first:
            self.sleep(self.spec.reobserve_seconds)
            try:
                return self.observe(self.spec.gpu_uuid), f"{type(first).__name__}: {first}"[:300]
            except _OBSERVATION_ERRORS as second:
                raise RuntimeError(f"GPU observation failed twice: {type(second).__name__}: {second}"[:400])

    def _unproven(self, observation):
        own = {os.getpid(), self.child.pid}
        try:
            own |= self.own_descendants(self.child.pid)
        except OSError:
            pass  # without the scan only the process group can prove ownership
        return unproven_pids(observation, self.child_pgid, own_pids=own, pgid_of=self.pgid_of)

    def watch_once(self):
        """Observe; re-observe once when a PID is not provably ours; foreign if it persists."""
        try:
            first, first_error = self._observe_twice_on_failure()
        except RuntimeError as error:
            return {"collision": True, "reason": "watchdog_observation_failed", "error": str(error), "foreign": []}
        if forbidden_gpu(first, self.spec.gpu_uuid):
            return {"collision": True, "reason": "gpu5", "observations": [first], "foreign": first["pids"]}
        unproven_first = self._unproven(first)
        if not unproven_first:
            return {"collision": False, "reason": None, "observations": [first], "foreign": [],
                    "observation_retry": first_error}
        self.sleep(self.spec.reobserve_seconds)
        try:
            second, _ = self._observe_twice_on_failure()
        except RuntimeError as error:
            return {"collision": True, "reason": "watchdog_observation_failed", "error": str(error),
                    "unproven_first": unproven_first, "foreign": unproven_first}
        if forbidden_gpu(second, self.spec.gpu_uuid):
            return {"collision": True, "reason": "gpu5", "observations": [first, second], "foreign": second["pids"]}
        unproven_second = self._unproven(second)
        foreign = sorted(set(unproven_first) & set(unproven_second))
        return {"collision": bool(foreign), "reason": "foreign_pid" if foreign else None,
                "observations": [first, second], "unproven_first": unproven_first,
                "unproven_second": unproven_second, "foreign": foreign}

    def _check_registry(self, models):
        rows = models.get("data") if isinstance(models, dict) else None
        return (isinstance(rows, list) and len(rows) == 1 and rows[0].get("id") == self.spec.served_model.alias
                and rows[0].get("max_model_len") == MAX_MODEL_LEN
                and rows[0].get("root") in (None, self.spec.model_dir))

    def run(self):
        spec = self.spec
        for directory in unit_directories(spec):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        check = (self.preflight_check or preflight)(spec, observe=self.observe)
        self.record("preflight", check)
        if not check["passed"]:
            return 2

        def stop(*_):
            self.stop_requested = True
        previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
        stamp = _utc()
        logs = [open(spec.cache_root / "logs" / f"vllm-{stamp}.{name}", "xb") for name in ("stdout", "stderr")]
        try:
            self.child = self.popen(vllm_argv(spec), stdin=subprocess.DEVNULL, stdout=logs[0], stderr=logs[1],
                                    env=serve_environment(spec), cwd=str(spec.cache_root), start_new_session=True)
            self.child_pgid = self.child.pid  # start_new_session: the child leads its own group
            self.record("launched", {"supervisor": process_identity(), "child": process_identity(self.child.pid),
                                     "pgid": self.child_pgid, "argv": vllm_argv(spec), "identity": current_identity(),
                                     "logs": [stream.name for stream in logs]})
            deadline = time.monotonic() + spec.startup_timeout_seconds
            last_watch = None
            ready = False
            while not self.stop_requested:
                if self.child.poll() is not None:
                    self.record("exited", {"phase": "running" if ready else "startup",
                                           "returncode": self.child.returncode})
                    return 1
                if last_watch is None or time.monotonic() - last_watch >= spec.watchdog_seconds:
                    last_watch = time.monotonic()
                    status = self.watch_once()
                    if status["collision"]:
                        self.record("collision", status)
                        self.stop_own_group("collision:" + status["reason"])
                        return 6 if status["reason"] == "watchdog_observation_failed" else 3
                    if status.get("unproven_first"):
                        self.record("watch_transient", status)
                if not ready:
                    if time.monotonic() > deadline:
                        self.stop_own_group("startup_timeout")
                        return 4
                    try:
                        self.http_get("/health")
                        models = self.http_get("/v1/models")
                    except (OSError, ValueError):
                        self.sleep(2)
                        continue
                    if not self._check_registry(models):
                        self.record("model_registry_mismatch", {"models": models})
                        self.stop_own_group("model_registry_mismatch")
                        return 5
                    ready = True
                    self.record("ready", {"endpoint": spec.endpoint, "models": models})
                self.sleep(1)
            self.stop_own_group("stop_requested")
            return 0
        finally:
            if self.child is not None and self.child.poll() is None:
                self.stop_own_group("supervisor_exit")
            for stream in logs:
                stream.close()
            for number, handler in previous.items():
                signal.signal(number, handler)
