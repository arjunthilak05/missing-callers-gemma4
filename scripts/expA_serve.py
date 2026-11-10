"""Exp A serving CLI (docs/EXPA_DESIGN.md section 7, docs/EXPA_RUNBOOK.md). The coordinator alone launches serving.

  gpus          read-only recheck of every GPU: used MiB, compute PIDs, idle and eligible (never GPU5)
  plan          print the vLLM argv, environment and the hardened `sudo systemd-run` command for one
                instance (one GPU, one port); writes and runs nothing; refuses unless run as thilak.arjun
  preflight     identity, GPU (refuses GPU5 by UUID, index and minor number), port, tmpfs cache root,
                serve env and model-file checks for one instance; exit 2 when anything fails
  verify-model  check the model files against the profile's pinned manifest (models/COPY.sha256 for the
                default gemma4-31b; full sha256 or quick), and a profile's chat-template override against its
                pinned sha256 (gemma4-31b-qat)
  supervise     the transient unit's entry point: preflight, launch vLLM in its own process group and
                watch the GPU every 30 s (a PID that is not provably ours on two consecutive
                observations stops only our own group and is recorded)

Every path defaults to the public mirror /mnt/kldata/absent-edges-expA-public-20261101 and can be overridden.
--model-profile (default gemma4-31b, the frozen Exp A model) selects the served alias, the revision and the pinned
model copy (serve.MODEL_PROFILES); --model-dir and --model-manifest default to that profile's copy. A profile's
chat-template override (gemma4-31b-qat: the frozen 31B BF16 template) is fixed by the profile, path and sha256.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from absent_edges.probe import serve


def _add_spec_arguments(command):
    command.add_argument("--run", required=True, help="run name; working dir /dev/shm/ae-expA-<run>/")
    command.add_argument("--instance", required=True, help="short id, e.g. g7; caches in <run dir>/serve/<id>/")
    command.add_argument("--gpu-uuid", required=True, help="full GPU UUID (never an index; never GPU5)")
    command.add_argument("--port", type=int, required=True, help="loopback port, one per instance")
    command.add_argument("--max-num-seqs", type=int, default=serve.DEFAULT_MAX_NUM_SEQS)
    command.add_argument("--gpu-memory-utilization", type=float, default=serve.DEFAULT_GPU_MEMORY_UTILIZATION)
    command.add_argument("--runtime-max-seconds", type=int, default=serve.SERVE_RUNTIME_MAX_SECONDS,
                         help="unit RuntimeMaxSec (default 6 days: covers the D11.3 deadlines)")
    command.add_argument("--memory-max", default="192G")
    command.add_argument("--serve-env", default=str(serve.SERVE_ENV))
    _add_model_arguments(command)
    command.add_argument("--runtime-root", default=str(serve.RUNTIME_ROOT),
                         help="the runtime Python tree the supervisor runs from (ownership is recorded)")
    command.add_argument("--mirror-owner-uid", type=int, default=None,
                         help="refuse unless every file and directory of the serve env, the runtime Python, the "
                              "model and their ancestors is owned by this UID (or root) and not group/other-writable "
                              "(0 once the public mirror is root-owned); default: record ownership only")
    command.add_argument("--model-check", choices=("full", "quick"), default="full",
                         help="full hashes every model byte before launch (default); quick hashes files <=256 MiB")


def _add_model_arguments(command):
    command.add_argument("--model-profile", choices=sorted(serve.MODEL_PROFILES), default=serve.DEFAULT_MODEL_PROFILE,
                         help="served model: alias, revision, pinned copy and required files (default %(default)s)")
    command.add_argument("--model-dir", default=None, help="default: the profile's copy in the public mirror")
    command.add_argument("--model-manifest", default=None,
                         help="default: the profile's sha256sum manifest (its own sha256 is pinned by the profile)")


def _spec(args):
    return serve.ServeSpec.for_profile(args.model_profile, run=args.run, instance=args.instance,
                                       gpu_uuid=args.gpu_uuid, port=args.port, max_num_seqs=args.max_num_seqs,
                                       gpu_memory_utilization=args.gpu_memory_utilization,
                                       runtime_max_seconds=args.runtime_max_seconds, memory_max=args.memory_max,
                                       serve_env=args.serve_env, model_dir=args.model_dir,
                                       model_manifest=args.model_manifest, runtime_root=args.runtime_root,
                                       mirror_owner_uid=args.mirror_owner_uid, model_check=args.model_check)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("gpus")
    for name in ("plan", "preflight"):
        command = commands.add_parser(name)
        _add_spec_arguments(command)
        if name == "plan":
            command.add_argument("--release-root", default=None,
                                 help="code release containing scripts/expA_serve.py "
                                      "(default: the staged copy <run dir>/inputs/release)")
            command.add_argument("--runtime-python", default=str(serve.RUNTIME_PYTHON),
                                 help="interpreter for the supervisor (stdlib only)")
    verify = commands.add_parser("verify-model")
    _add_model_arguments(verify)
    verify.add_argument("--mode", choices=("full", "quick"), default="full")
    supervise = commands.add_parser("supervise")
    supervise.add_argument("--spec", required=True, help="JSON ServeSpec (as printed by plan)")
    args = parser.parse_args(argv)
    if args.command == "gpus":
        print(json.dumps(serve.gpu_table(), indent=2, sort_keys=True))
        return 0
    if args.command == "plan":
        try:
            result = serve.plan(_spec(args), release_root=args.release_root, runtime_python=args.runtime_python)
        except serve.ServePreflightError as error:
            print(json.dumps({"refused": str(error)}, sort_keys=True))
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = serve.preflight(_spec(args))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 2
    if args.command == "verify-model":
        served = serve.served_model(args.model_profile)
        result = serve.verify_model(args.model_dir or served.model_dir, args.model_manifest or served.manifest,
                                    mode=args.mode, expected_manifest_sha256=served.manifest_sha256,
                                    required_files=served.required_files)
        if served.chat_template is not None:
            template = serve.verify_chat_template(served.chat_template, served.chat_template_sha256)
            result["chat_template_override"] = template
            result["failures"] = result["failures"] + template["failures"]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not result["failures"] else 2
    spec = serve.ServeSpec(**json.loads(args.spec))
    return serve.Supervisor(spec).run()


if __name__ == "__main__":
    raise SystemExit(main())
