#!/usr/bin/env bash
# Exp A operator sequence on Core1 (ai-core-team-1). See docs/EXPA_RUNBOOK.md for the full procedure.
#
# Run it as thilak.arjun (Core1 UID 1008) from the coordinator's checkout, without copying it to Core1:
#   ssh -o BatchMode=yes ai-core-team-1 'bash -c "$(cat)" expA_core1_launch <command> [options]' \
#       < scripts/expA_core1_launch.sh
#
# Commands (every step after prepare takes --run NAME and reads /dev/shm/ae-expA-NAME/control/launch.env):
#   gpus            read-only GPU recheck (used MiB, compute PIDs, idle, GPU5 flagged)
#   prepare         create /dev/shm/ae-expA-NAME (0700) and its control/launch.env from the options below
#   add-instance    add a replacement GPU instance NAME=GPU-UUID=PORT to a prepared run
#   stage           root stager (hardened one-shot unit): release, dispatch view, clean checkouts -> inputs/, and
#                   the run-state check against the Core2 run dir (--restore after a reboot wiped /dev/shm)
#   serve           per instance: GPU recheck, serve plan from the staged release, hardened vLLM unit
#   serve-status    units, supervisor records and /v1/models of every instance
#   runner          the queue runner as thilak.arjun in a hardened unit, one --endpoint per instance
#   publisher       root publisher: init the Core2 run dir, then the hardened publish loop unit
#   status          units, GPUs, run status from durable records, last publisher passes
#   drain           SIGTERM to the runner's main process only: finish in-flight episodes, start no new ones
#   stop-runner     stop the runner unit (SIGTERM = drain, SIGKILL after TimeoutStopSec)
#   stop-serve      stop serving units (all, or --instance NAME)
#   final-publish   (runner and serve units stopped) reconcile closes abandoned attempts -> stop file -> the
#                   publish loop exits -> one hardened final pass with logs; exit 5 = records still waiting
#   cleanup         remove /dev/shm/ae-expA-NAME after a clean final publish (--confirm)
#   show-config     print control/launch.env
#
# Model and budgets: prepare --model-profile gemma4-31b|gemma4-12b|gemma4-31b-qat (default gemma4-31b, the frozen
# Exp A model) is recorded in launch.env as MODEL_PROFILE; serve plans that profile (gemma4-31b-qat adds the frozen 31B
# --chat-template, checked by the supervisor) and the runner requires its alias. runner
# --max-model-calls N --context-budget TOKENS (defaults 8 and 16384, the prereg values) set the episode budgets. runner
# --final-turn-policy none|neutral|deadline (default none, protocols/EXPA_FINALTURN.md) or deadline_info|submit_only
# (protocols/EXPA_A4_DIRECTIVE.md) appends the notice before each episode's last response; every policy but none is
# forwarded only with the default budget and model profile and with --blocks P and one seed, --seeds 11 or --seeds 23
# (the protocols' cells), optionally with --conditions from C0,C1. runner --conditions LIST (any run) is forwarded as a
# selection filter like --seeds. runner --evidence-ledger none|off|on (default none,
# protocols/EXPA_A3_LEDGER.md) runs an A3 arm; off and on are forwarded only with --final-turn-policy deadline, the
# default budget and model profile, and --blocks LEDGER (seeds from 61, 73, 89, or none for all three). runner
# --submit-review none|neutral|ledger (default none, protocols/EXPA_A5_REVIEW.md) runs an A5 arm; neutral and ledger
# are forwarded only with --final-turn-policy deadline, no --evidence-ledger (or none), the default budget and model
# profile, and --blocks LEDGER (seeds as A3). Follow-up A6 (protocols/EXPA_A6_BUDGET16.md) is the one exception to the
# default budget: runner --final-turn-policy deadline --evidence-ledger off --max-model-calls 16 --context-budget 32768
# is forwarded with the default model profile and --blocks LEDGER (seeds as A3). runner --task-framing default|impact
# (default default, protocols/EXPA_A7_IMPACT.md) runs follow-up A7; impact is forwarded only with the default budget
# and model profile, no --final-turn-policy, --evidence-ledger or --submit-review (or none), and exactly --blocks P,S-VC
# --seeds 11 --conditions C0,C1 (on a fresh run the order guard also needs --allow-out-of-order REASON). With the
# defaults, the serve and runner units are started with exactly the commands of the frozen release.
#
# Never GPU5. Never another person's UID. Never /data5t. SQLite and locks stay on /dev/shm. No timers.
set -euo pipefail
umask 077
if [ -n "${BASH_EXECUTION_STRING:-}" ]; then exec </dev/null; fi

EXPECTED_USER=thilak.arjun
EXPECTED_UID=1008
EXPECTED_HOST=ai-core-team-1
GPU5_UUID=GPU-7b8d9f89-0a8f-2d07-4833-bee70a0eeed2
PUBLIC=/mnt/kldata/absent-edges-expA-public-20261101
RUNTIME_PYTHON_DEFAULT=$PUBLIC/runtime/python3.11/bin/python3.11
SYSTEM_PYTHON=/usr/bin/python3
TARGET_PARENT_DEFAULT=/mnt/kldata/absent-edges-20261018-core2/expA/runs
MODEL_PROFILE_DEFAULT=gemma4-31b
MODEL_ALIAS=absent-edges-gemma4-31b-bf16     # load_config sets it from MODEL_PROFILE
INACCESSIBLE="-/data5t -/workspace -/mnt/eu-ai-datasets"
ROOT_CAPS="CAP_CHOWN CAP_DAC_OVERRIDE CAP_DAC_READ_SEARCH CAP_FOWNER"
# The sha256 loader (identical to scripts/expA_publish.py LOADER): root executes exactly the pinned bytes.
LOADER="import hashlib,sys;p,h=sys.argv[1],sys.argv[2];b=open(p,'rb').read();d=hashlib.sha256(b).hexdigest();d==h or sys.exit('ae-expA loader: sha256 '+d+' of '+p+' is not the pinned '+h);sys.argv=[p]+sys.argv[3:];exec(compile(b,p,'exec'),{'__name__':'__main__','__file__':p,'__loader_sha256__':d,'__builtins__':__builtins__})"
CONFIG_KEYS="RUN RUN_ID RELEASE_DIR RELEASE_SUMS_SHA256 STAGER_SHA256 DISPATCH_DIR DISPATCH_MANIFEST_SHA256 \
CHECKOUTS_MANIFEST CHECKOUTS_MANIFEST_SHA256 CHECKOUTS_DIR INSTANCES CONCURRENCY GIT_SHA TARGET_PARENT \
RUNTIME_PYTHON MAX_NUM_SEQS GPU_MEMORY_UTILIZATION MODEL_CHECK MIRROR_OWNER_UID MODEL_PROFILE"
# ProtectSystem=strict leaves the API file systems (/dev, so /dev/shm) writable: every unit makes /dev/shm
# read-only and reopens only its own directories with the more specific ReadWritePaths=.
SHM_READ_ONLY="ReadOnlyPaths=/dev/shm"

die() { printf 'expA_core1_launch: ERROR: %s\n' "$*" >&2; exit 2; }
info() { printf 'expA_core1_launch: %s\n' "$*" >&2; }

require_identity() {
    [ "$(id -un)" = "$EXPECTED_USER" ] || die "run as $EXPECTED_USER, not $(id -un)"
    [ "$(id -u)" = "$EXPECTED_UID" ] || die "$EXPECTED_USER must be UID $EXPECTED_UID here, found $(id -u)"
    [ "$(hostname -s)" = "$EXPECTED_HOST" ] || die "this script runs on $EXPECTED_HOST, not $(hostname -s)"
}

match() { [[ "$2" =~ $1 ]]; }
check_name() { match '^[a-z0-9][a-z0-9-]{0,47}$' "$1" || die "invalid run or instance name: $1"; }
check_run_id() { match '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$' "$1" || die "invalid run id: $1"; }
check_sha() { match '^[0-9a-f]{64}$' "$1" || die "$2 must be 64 lowercase hex digits"; }
check_path() { match '^/[A-Za-z0-9_./@+-]+$' "$1" || die "$2 must be an absolute path without spaces: $1"; }
check_int() { match '^[1-9][0-9]{0,5}$' "$1" || die "$2 must be a positive integer: $1"; }
check_uuid() {
    match '^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' "$1" || die "invalid GPU UUID: $1"
    [ "$1" != "$GPU5_UUID" ] || die "GPU5 is never used"
}

run_root() { printf '/dev/shm/ae-expA-%s' "$1"; }

model_alias() {  # PROFILE -> the alias it serves (absent_edges/probe/serve.py MODEL_PROFILES; serve checks the plan)
    case $1 in
        gemma4-31b) printf '%s' absent-edges-gemma4-31b-bf16 ;;
        gemma4-12b) printf '%s' absent-edges-gemma4-12b-bf16 ;;
        gemma4-31b-qat) printf '%s' absent-edges-gemma4-31b-qat-w4a16 ;;
        *) return 1 ;;
    esac
}

load_config() {
    local run=$1 file line key value
    check_name "$run"
    ROOT=$(run_root "$run")
    file=$ROOT/control/launch.env
    [ -f "$file" ] && [ ! -L "$file" ] || die "no $file; run prepare first"
    [ "$(stat -c '%u %a' "$ROOT")" = "$EXPECTED_UID 700" ] || die "$ROOT must be 0700 and owned by UID $EXPECTED_UID"
    for key in $CONFIG_KEYS; do printf -v "$key" '%s' ""; done
    while IFS= read -r line || [ -n "$line" ]; do
        [ -z "$line" ] && continue
        key=${line%%=*}
        value=${line#*=}
        case " $CONFIG_KEYS " in *" $key "*) ;; *) die "unknown key $key in $file" ;; esac
        match '^[A-Za-z0-9_./:=,@+ -]*$' "$value" || die "unsafe value for $key in $file"
        printf -v "$key" '%s' "$value"
    done < "$file"
    [ "$RUN" = "$run" ] || die "$file belongs to run $RUN"
    MODEL_PROFILE=${MODEL_PROFILE:-$MODEL_PROFILE_DEFAULT}    # a launch.env written before the option: the 31B
    MODEL_ALIAS=$(model_alias "$MODEL_PROFILE") || die "unknown model profile $MODEL_PROFILE in $file"
    PY=${RUNTIME_PYTHON:-$RUNTIME_PYTHON_DEFAULT}
    REL=$ROOT/inputs/release
    CONTROL=$ROOT/control
}

each_instance() {  # prints "name gpu port" lines
    local spec
    for spec in $INSTANCES; do
        IFS='=' read -r name gpu port <<<"$spec"
        printf '%s %s %s\n' "$name" "$gpu" "$port"
    done
}

gpu_report() {  # read-only
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits
}

gpu_check() {  # UUID: idle (<=128 MiB, no compute process) and not GPU5 by UUID or index
    local uuid=$1 row index used apps
    check_uuid "$uuid"
    row=$(gpu_report | awk -F', *' -v u="$uuid" '$2==u {print $1" "$3}')
    [ -n "$row" ] || die "GPU $uuid not found by nvidia-smi"
    read -r index used <<<"$row"
    [ "$index" != "5" ] || die "GPU $uuid has nvidia-smi index 5; GPU5 is never used"
    [ "$used" -le 128 ] || die "GPU $uuid has $used MiB in use"
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', *' -v u="$uuid" '$1==u' | wc -l)
    [ "$apps" -eq 0 ] || die "GPU $uuid has $apps compute processes"
    info "GPU $uuid (index $index) is idle: $used MiB, no compute processes"
}

verify_release() {  # the staged release must be the pinned one; prints nothing
    [ -f "$REL/RELEASE_FILES.sha256" ] || die "no staged release; run stage first"
    [ "$(sha256sum "$REL/RELEASE_FILES.sha256" | cut -c1-64)" = "$RELEASE_SUMS_SHA256" ] \
        || die "staged RELEASE_FILES.sha256 differs from the pinned $RELEASE_SUMS_SHA256"
    (cd "$REL" && sha256sum --quiet --strict -c RELEASE_FILES.sha256) || die "staged release files differ from their sums"
}

release_sha() {  # sha256 of one release file from the staged, pinned RELEASE_FILES.sha256
    awk -v f="$1" '$2==f || $2=="./"f {print $1}' "$REL/RELEASE_FILES.sha256"
}

json_argv() {  # FILE KEY... -> NUL-separated argv list found at that key path
    "$PY" -I -B -c 'import json,sys
v=json.load(open(sys.argv[1]))
for k in sys.argv[2:]: v=v[k]
assert isinstance(v,list) and all(isinstance(x,str) for x in v)
sys.stdout.write("\0".join(v))' "$@"
}

endpoint_ready() {  # PORT -> 0 when /v1/models serves exactly the alias with max_model_len 32768
    curl -s --noproxy '*' --max-time 5 "http://127.0.0.1:$1/v1/models" | "$PY" -I -B -c 'import json,sys
d=json.load(sys.stdin)["data"]
sys.exit(0 if len(d)==1 and d[0].get("id")==sys.argv[1] and d[0].get("max_model_len")==32768 else 1)' "$MODEL_ALIAS"
}

# ---------------------------------------------------------------------------------------------------------

cmd_gpus() {
    require_identity
    printf 'index, uuid, used_mib, compute_pids, verdict\n'
    local apps index uuid used pids verdict
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader || true)
    while IFS=', ' read -r index uuid used; do
        pids=$(printf '%s\n' "$apps" | awk -F', *' -v u="$uuid" '$1==u {printf "%s ", $2}')
        verdict=idle
        [ "$used" -le 128 ] && [ -z "$pids" ] || verdict=busy
        { [ "$uuid" = "$GPU5_UUID" ] || [ "$index" = 5 ]; } && verdict="$verdict,GPU5-NEVER"
        printf '%s, %s, %s, %s, %s\n' "$index" "$uuid" "$used" "${pids:--}" "$verdict"
    done < <(gpu_report)
}

cmd_prepare() {
    require_identity
    local run="" run_id="" release_dir="" release_sums="" stager_sha="" dispatch_dir="" dispatch_sha=""
    local checkouts_manifest="" checkouts_sha="" checkouts_dir="" instances="" concurrency=8 git_sha=""
    local target_parent=$TARGET_PARENT_DEFAULT runtime_python=$RUNTIME_PYTHON_DEFAULT max_num_seqs=8
    local gpu_util=0.92 model_check=full mirror_owner_uid="" model_profile=$MODEL_PROFILE_DEFAULT
    while [ $# -gt 0 ]; do
        case $1 in
            --run) run=$2 ;; --run-id) run_id=$2 ;; --release-dir) release_dir=$2 ;;
            --release-sums-sha256) release_sums=$2 ;; --stager-sha256) stager_sha=$2 ;;
            --dispatch-dir) dispatch_dir=$2 ;; --dispatch-manifest-sha256) dispatch_sha=$2 ;;
            --checkouts-manifest) checkouts_manifest=$2 ;; --checkouts-manifest-sha256) checkouts_sha=$2 ;;
            --checkouts-dir) checkouts_dir=$2 ;; --instance) instances="$instances $2" ;;
            --concurrency) concurrency=$2 ;; --git-sha) git_sha=$2 ;; --target-parent) target_parent=$2 ;;
            --runtime-python) runtime_python=$2 ;; --max-num-seqs) max_num_seqs=$2 ;;
            --gpu-memory-utilization) gpu_util=$2 ;; --model-check) model_check=$2 ;;
            --mirror-owner-uid) mirror_owner_uid=$2 ;; --model-profile) model_profile=$2 ;;
            *) die "prepare: unknown option $1" ;;
        esac
        shift 2
    done
    [ -z "$mirror_owner_uid" ] || match '^(0|[1-9][0-9]{0,6})$' "$mirror_owner_uid" \
        || die "--mirror-owner-uid must be a UID (0 once the public mirror chain is root-owned)"
    model_alias "$model_profile" > /dev/null \
        || die "--model-profile must be gemma4-31b-qat or gemma4-31b or gemma4-12b, not $model_profile"
    check_name "$run"
    run_id=${run_id:-$run}
    check_run_id "$run_id"
    check_path "$release_dir" --release-dir
    check_sha "$release_sums" --release-sums-sha256
    check_sha "$stager_sha" --stager-sha256
    check_path "$dispatch_dir" --dispatch-dir
    check_sha "$dispatch_sha" --dispatch-manifest-sha256
    check_path "$checkouts_manifest" --checkouts-manifest
    check_sha "$checkouts_sha" --checkouts-manifest-sha256
    [ -z "$checkouts_dir" ] || check_path "$checkouts_dir" --checkouts-dir
    check_path "$target_parent" --target-parent
    check_path "$runtime_python" --runtime-python
    check_int "$concurrency" --concurrency
    check_int "$max_num_seqs" --max-num-seqs
    match '^0\.[0-9]{1,3}$' "$gpu_util" || die "--gpu-memory-utilization must look like 0.92"
    match '^(full|quick)$' "$model_check" || die "--model-check must be full or quick"
    match '^[0-9a-f]{7,40}$' "$git_sha" || die "--git-sha (the release commit) is required"
    instances=${instances# }
    [ -n "$instances" ] || die "at least one --instance NAME=GPU-UUID=PORT is required"
    local spec seen=""
    for spec in $instances; do
        validate_instance "$spec" "$seen"
        seen="$seen $spec"
    done
    ROOT=$(run_root "$run")
    if [ -e "$ROOT" ]; then
        [ -d "$ROOT" ] && [ ! -L "$ROOT" ] && [ "$(stat -c '%u %a' "$ROOT")" = "$EXPECTED_UID 700" ] \
            || die "$ROOT exists and is not our 0700 directory"
        [ ! -e "$ROOT/control/launch.env" ] || die "$ROOT/control/launch.env exists; a run is prepared once"
    fi
    install -d -m 0700 "$ROOT" "$ROOT/control" "$ROOT/logs" "$ROOT/serve" "$ROOT/run" "$ROOT/ledger" \
        "$ROOT/scratch" "$ROOT/scratch/home" "$ROOT/scratch/tmp"
    [ "$(findmnt -no FSTYPE --target "$ROOT")" = tmpfs ] || die "$ROOT is not on tmpfs"
    {
        printf 'RUN=%s\nRUN_ID=%s\nRELEASE_DIR=%s\nRELEASE_SUMS_SHA256=%s\nSTAGER_SHA256=%s\n' \
            "$run" "$run_id" "$release_dir" "$release_sums" "$stager_sha"
        printf 'DISPATCH_DIR=%s\nDISPATCH_MANIFEST_SHA256=%s\nCHECKOUTS_MANIFEST=%s\nCHECKOUTS_MANIFEST_SHA256=%s\n' \
            "$dispatch_dir" "$dispatch_sha" "$checkouts_manifest" "$checkouts_sha"
        printf 'CHECKOUTS_DIR=%s\nINSTANCES=%s\nCONCURRENCY=%s\nGIT_SHA=%s\nTARGET_PARENT=%s\n' \
            "$checkouts_dir" "$instances" "$concurrency" "$git_sha" "$target_parent"
        printf 'RUNTIME_PYTHON=%s\nMAX_NUM_SEQS=%s\nGPU_MEMORY_UTILIZATION=%s\nMODEL_CHECK=%s\n' \
            "$runtime_python" "$max_num_seqs" "$gpu_util" "$model_check"
        printf 'MIRROR_OWNER_UID=%s\n' "$mirror_owner_uid"
        printf 'MODEL_PROFILE=%s\n' "$model_profile"
    } > "$ROOT/control/launch.env.tmp"
    chmod 0400 "$ROOT/control/launch.env.tmp"
    mv -n "$ROOT/control/launch.env.tmp" "$ROOT/control/launch.env"
    findmnt -no OPTIONS /proc > "$ROOT/control/proc-mount-options.txt" || true
    info "prepared $ROOT"
    cat "$ROOT/control/launch.env"
}

validate_instance() {  # SPEC EXISTING_INSTANCES
    local name gpu port other_name other_gpu other_port entry
    IFS='=' read -r name gpu port <<<"$1"
    check_name "$name"; check_uuid "$gpu"; check_int "$port" "port of $name"
    [ "$port" -ge 1024 ] && [ "$port" -le 65535 ] || die "port of $name must be 1024-65535"
    for entry in $2; do
        IFS='=' read -r other_name other_gpu other_port <<<"$entry"
        [ "$name" != "$other_name" ] || die "duplicate instance $name"
        [ "$gpu" != "$other_gpu" ] || die "GPU $gpu appears twice (one vLLM instance per GPU)"
        [ "$port" != "$other_port" ] || die "port $port appears twice"
    done
}

cmd_add_instance() {  # a replacement GPU within the same run (the frozen design and run id stay the same)
    require_identity
    local run="" spec=""
    while [ $# -gt 0 ]; do
        case $1 in --run) run=$2 ;; --instance) spec=$2 ;; *) die "add-instance: unknown option $1" ;; esac
        shift 2
    done
    load_config "$run"
    validate_instance "$spec" "$INSTANCES"
    sed "s|^INSTANCES=.*|INSTANCES=$INSTANCES $spec|" "$CONTROL/launch.env" > "$CONTROL/launch.env.tmp"
    chmod 0400 "$CONTROL/launch.env.tmp"
    mv -f "$CONTROL/launch.env.tmp" "$CONTROL/launch.env"
    info "added instance $spec to $RUN"
    grep '^INSTANCES=' "$CONTROL/launch.env"
}

cmd_stage() {
    require_identity
    local run="" restore=()
    while [ $# -gt 0 ]; do
        case $1 in --run) run=$2; shift 2 ;; --restore) restore=(--restore); shift ;; *) die "stage: unknown option $1" ;; esac
    done
    load_config "$run"
    local extra=()
    [ -z "$CHECKOUTS_DIR" ] || extra=(--checkouts-dir "$CHECKOUTS_DIR")
    info "staging into $ROOT/inputs as root (hardened one-shot unit; reads Core2 through NFS)"
    # run-state: compares the records already published to $TARGET_PARENT/$RUN_ID with $ROOT/run before any
    # runner launch; refuses an empty run/ when Core2 holds records (a reboot wiped /dev/shm) unless --restore.
    sudo -n systemd-run --unit="ae-expA-stage-$RUN" --description="Absent Edges Exp A stager $RUN" \
        --wait --pipe --collect --property=Type=exec --property=User=root --property=NoNewPrivileges=yes \
        --property=ProtectSystem=strict --property=ProtectHome=yes --property=PrivateTmp=yes \
        --property=PrivateNetwork=yes --property=ProtectKernelTunables=yes --property=ProtectKernelModules=yes \
        --property=ProtectControlGroups=yes --property=RestrictSUIDSGID=yes --property=UMask=0077 \
        --property="CapabilityBoundingSet=$ROOT_CAPS" --property="InaccessiblePaths=$INACCESSIBLE" \
        --property="$SHM_READ_ONLY" --property="ReadWritePaths=$ROOT" \
        --setenv=LANG=C.UTF-8 --setenv=PYTHONDONTWRITEBYTECODE=1 -- \
        "$SYSTEM_PYTHON" -I -B -c "$LOADER" "$RELEASE_DIR/scripts/expA_stage.py" "$STAGER_SHA256" \
        --run-root "$ROOT" --owner "$EXPECTED_USER" --components release,dispatch,checkouts,run-state \
        --release-dir "$RELEASE_DIR" --expect-release-sums-sha256 "$RELEASE_SUMS_SHA256" \
        --dispatch-dir "$DISPATCH_DIR" --expect-dispatch-manifest-sha256 "$DISPATCH_MANIFEST_SHA256" \
        --checkouts-manifest "$CHECKOUTS_MANIFEST" --expect-checkouts-manifest-sha256 "$CHECKOUTS_MANIFEST_SHA256" \
        --run-target "$TARGET_PARENT/$RUN_ID" "${restore[@]}" "${extra[@]}"
    verify_release
    [ "$(release_sha scripts/expA_stage.py)" = "$STAGER_SHA256" ] \
        || die "the stager that ran is not the release's scripts/expA_stage.py"
    local bad
    bad=$(find "$ROOT/inputs" \( -type d ! -perm 0700 \) -o \( -type f ! -perm 0444 \) -o ! -uid "$EXPECTED_UID" | head -5)
    [ -z "$bad" ] || die "staged inputs with unexpected owner or mode: $bad"
    "$PY" -I -B -c 'import sys; sys.path.insert(0, sys.argv[1]); import absent_edges.probe.harness, absent_edges.probe.tools, absent_edges.probe.serve, absent_edges.graphs' "$REL" \
        || die "the runtime Python cannot import the staged harness"
    info "staged and verified: $(find "$ROOT/inputs" -type f | wc -l) files"
}

cmd_serve() {
    require_identity
    local run="" only=""
    while [ $# -gt 0 ]; do
        case $1 in --run) run=$2 ;; --instance) only="$only $2" ;; *) die "serve: unknown option $1" ;; esac
        shift 2
    done
    load_config "$run"
    verify_release
    local name gpu port plan argv owner=()
    [ -z "${MIRROR_OWNER_UID:-}" ] || owner=(--mirror-owner-uid "$MIRROR_OWNER_UID")
    while read -r name gpu port; do
        if [ -n "$only" ]; then case " $only " in *" $name "*) ;; *) continue ;; esac; fi
        if systemctl is-active --quiet "ae-expA-serve-$RUN-$name"; then info "$name already active"; continue; fi
        # A unit that failed before --collect existed stays loaded and blocks its name: clear it (no-op otherwise).
        sudo -n systemctl reset-failed "ae-expA-serve-$RUN-$name" 2>/dev/null || true
        gpu_check "$gpu"
        plan=$CONTROL/serve-$name.plan.json
        "$PY" -I -B "$REL/scripts/expA_serve.py" plan --run "$RUN" --instance "$name" --gpu-uuid "$gpu" \
            --port "$port" --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --model-check "$MODEL_CHECK" --release-root "$REL" --runtime-python "$PY" "${owner[@]}" \
            --model-profile "$MODEL_PROFILE" > "$plan.tmp"
        mv -f "$plan.tmp" "$plan"
        "$PY" -I -B -c 'import json,sys
m=json.load(open(sys.argv[1]))["model"]
sys.exit(0 if [m.get("profile"),m.get("alias")]==sys.argv[2:] else 1)' "$plan" "$MODEL_PROFILE" "$MODEL_ALIAS" \
            || die "the serve plan of $name is not model profile $MODEL_PROFILE ($MODEL_ALIAS)"
        install -d -m 0700 "$ROOT/serve/$name"
        mapfile -d '' -t argv < <(json_argv "$plan" systemd_run_argv)
        [ "${argv[0]}" = systemd-run ] || die "unexpected serve plan for $name"
        info "starting ae-expA-serve-$RUN-$name on $gpu port $port"
        sudo -n "${argv[@]}"
    done < <(each_instance)
    info "the supervisor hashes the $MODEL_PROFILE model (MODEL_CHECK=$MODEL_CHECK), then starts vLLM as $MODEL_ALIAS;"\
" watch with serve-status"
}

cmd_serve_status() {
    require_identity
    local run=""
    while [ $# -gt 0 ]; do case $1 in --run) run=$2; shift 2 ;; *) die "serve-status: unknown option $1" ;; esac; done
    load_config "$run"
    local name gpu port
    while read -r name gpu port; do
        printf '== %s %s port %s\n' "$name" "$gpu" "$port"
        systemctl show "ae-expA-serve-$RUN-$name" -p ActiveState,SubState,Result,MainPID --no-pager || true
        ls "$ROOT/serve/$name/records" 2>/dev/null | tail -5 || true
        if endpoint_ready "$port"; then echo "endpoint: ready"; else echo "endpoint: not ready"; fi
    done < <(each_instance)
    gpu_report
}

cmd_runner() {
    require_identity
    local run="" blocks="" seeds="" families="" limit="" concurrency="" only="" out_of_order=""
    local max_calls="" context_budget="" final_turn="" ledger="" conditions="" review="" framing=""
    while [ $# -gt 0 ]; do
        case $1 in
            --run) run=$2 ;; --blocks) blocks=$2 ;; --seeds) seeds=$2 ;; --families) families=$2 ;;
            --conditions) conditions=$2 ;;
            --limit) limit=$2 ;; --concurrency) concurrency=$2 ;; --instances) only=${2//,/ } ;;
            --allow-out-of-order) out_of_order=$2 ;;
            --max-model-calls) max_calls=$2 ;; --context-budget) context_budget=$2 ;;
            --final-turn-policy) final_turn=$2 ;;
            --evidence-ledger) ledger=$2 ;;
            --submit-review) review=$2 ;;
            --task-framing) framing=$2 ;;
            *) die "runner: unknown option $1" ;;
        esac
        shift 2
    done
    load_config "$run"
    verify_release
    concurrency=${concurrency:-$CONCURRENCY}
    check_int "$concurrency" --concurrency
    local unit=ae-expA-runner-$RUN endpoints=() name gpu port
    ! systemctl is-active --quiet "$unit" || die "$unit is already active"
    # The stager's run-state check (published Core2 records vs run/) must have passed on this boot.
    compgen -G "$ROOT/inputs/STAGE_RECEIPT.run-state.*.json" > /dev/null \
        || die "no run-state receipt in $ROOT/inputs: run stage (it checks the Core2 run directory) first"
    while read -r name gpu port; do
        if [ -n "$only" ]; then case " $only " in *" $name "*) ;; *) continue ;; esac; fi
        endpoint_ready "$port" || die "instance $name (port $port) does not serve $MODEL_ALIAS yet"
        endpoints+=(--endpoint "http://127.0.0.1:$port=$gpu")
    done < <(each_instance)
    [ "${#endpoints[@]}" -gt 0 ] || die "no instance selected"
    local selection=()
    [ -z "$blocks" ] || { match '^[A-Z][A-Z,-]*$' "$blocks" || die "bad --blocks"; selection+=(--blocks "$blocks"); }
    [ -z "$seeds" ] || { match '^[0-9,]+$' "$seeds" || die "bad --seeds"; selection+=(--seeds "$seeds"); }
    [ -z "$families" ] || { match '^[A-Za-z0-9._,-]+$' "$families" || die "bad --families"; selection+=(--families "$families"); }
    [ -z "$conditions" ] || { match '^[A-Za-z0-9,-]+$' "$conditions" || die "bad --conditions"; selection+=(--conditions "$conditions"); }
    [ -z "$limit" ] || { check_int "$limit" --limit; selection+=(--limit "$limit"); }
    [ -z "$out_of_order" ] || { match '^[A-Za-z0-9._:-]{3,200}$' "$out_of_order" \
        || die "--allow-out-of-order needs a reason without spaces (use hyphens)"
        selection+=(--allow-out-of-order "$out_of_order"); }
    # Only a non-default profile or budget adds arguments: a default runner unit runs the frozen command. run.json
    # freezes both, so every relaunch of a run must repeat the same --max-model-calls and --context-budget.
    local model=() budget=()
    [ "$MODEL_PROFILE" = "$MODEL_PROFILE_DEFAULT" ] || model=(--model-profile "$MODEL_PROFILE")
    [ -z "$max_calls" ] || { check_int "$max_calls" --max-model-calls; budget+=(--max-model-calls "$max_calls"); }
    [ -z "$context_budget" ] || { check_int "$context_budget" --context-budget
        [ "$context_budget" -le 32768 ] || die "--context-budget must be at most 32768 (the served max_model_len)"
        budget+=(--context-budget "$context_budget"); }
    # Registered designs only: the prereg default (neither option) or 16/32768 (protocols/EXPA_BUDGET16.md).
    case "$max_calls/$context_budget" in
        /|16/32768) ;;
        *) die "--max-model-calls/--context-budget: only the registered 16/32768 design (protocols/EXPA_BUDGET16.md), or neither" ;;
    esac
    # A final-turn policy (protocols/EXPA_FINALTURN.md; deadline_info and submit_only: protocols/EXPA_A4_DIRECTIVE.md):
    # none (or no option) is the prereg harness and adds nothing; every other policy is registered only with the
    # default budget and model profile, for block P at one seed, 11 or 23, optionally restricted to conditions from
    # C0,C1 (run.json freezes the whole design, so a relaunch without that selection would dispatch unregistered
    # cells). run.json freezes the policy too. Under an A3 arm (below) the A3 selection replaces block P.
    # A6 (protocols/EXPA_A6_BUDGET16.md) is the one notice run at the 16/32768 budget: deadline with --evidence-ledger
    # off (A3's off arm) on block LEDGER, with the default model profile.
    local policy=() protocol=protocols/EXPA_FINALTURN.md a6="" a6_hint=""
    local a6_protocol=protocols/EXPA_A6_BUDGET16.md
    [ "$max_calls/$context_budget" != 16/32768 ] \
        || a6_hint="; with --max-model-calls 16 --context-budget 32768 only as A6: --final-turn-policy deadline --evidence-ledger off --blocks LEDGER and model profile $MODEL_PROFILE_DEFAULT ($a6_protocol)"
    case "$final_turn" in
        ""|none) ;;
        neutral|deadline|deadline_info|submit_only)
            case "$final_turn" in deadline_info|submit_only) protocol=protocols/EXPA_A4_DIRECTIVE.md ;; esac
            if [ "$final_turn" = deadline ] && [ "$max_calls/$context_budget" = 16/32768 ] && [ "$ledger" = off ]; then
                [ "$MODEL_PROFILE" = "$MODEL_PROFILE_DEFAULT" ] \
                    || die "--final-turn-policy deadline --evidence-ledger off with --max-model-calls 16 --context-budget 32768: registered only with model profile $MODEL_PROFILE_DEFAULT ($a6_protocol)"
                a6=1
            else
                local hint=""
                [ "$final_turn" != deadline ] || hint=$a6_hint
                [ -z "$max_calls$context_budget" ] && [ "$MODEL_PROFILE" = "$MODEL_PROFILE_DEFAULT" ] \
                    || die "--final-turn-policy $final_turn: registered only with the default budget (8 responses, 16384 tokens) and model profile $MODEL_PROFILE_DEFAULT ($protocol)$hint"
            fi
            if { [ -z "$ledger" ] || [ "$ledger" = none ]; } && { [ -z "$review" ] || [ "$review" = none ]; }; then
                [ "$blocks" = P ] && { [ "$seeds" = 11 ] || [ "$seeds" = 23 ]; } \
                    && { [ -z "$conditions" ] || match '^(C0|C1)(,(C0|C1))*$' "$conditions"; } \
                    || die "--final-turn-policy $final_turn: block P at seed 11 or 23 only (one seed per launch), pass --blocks P --seeds 11 or --seeds 23, optionally --conditions from C0,C1 ($protocol)"
            fi
            policy=(--final-turn-policy "$final_turn") ;;
        *) die "--final-turn-policy must be none, neutral, deadline, deadline_info or submit_only, not $final_turn" ;;
    esac
    # An A3 arm (protocols/EXPA_A3_LEDGER.md): none (or no option) adds nothing; off and on are registered only with
    # --final-turn-policy deadline, the default budget and model profile, for block LEDGER (run.json freezes the A3
    # design, whose other blocks are never dispatched under the ledger option). A6 (set above: deadline, off and the
    # 16/32768 budget) passes the same block LEDGER rule under its own protocol.
    local ledger_arm=() ledger_protocol=protocols/EXPA_A3_LEDGER.md
    case "$ledger" in
        ""|none) ;;
        off|on)
            if [ -n "$a6" ]; then
                ledger_protocol=$a6_protocol
            else
                [ "$final_turn" = deadline ] && [ -z "$max_calls$context_budget" ] && [ "$MODEL_PROFILE" = "$MODEL_PROFILE_DEFAULT" ] \
                    || die "--evidence-ledger $ledger: registered only with --final-turn-policy deadline, the default budget (8 responses, 16384 tokens) and model profile $MODEL_PROFILE_DEFAULT (protocols/EXPA_A3_LEDGER.md)$a6_hint"
            fi
            [ "$blocks" = LEDGER ] && { [ -z "$seeds" ] || match '^(61|73|89)(,(61|73|89))*$' "$seeds"; } \
                || die "--evidence-ledger $ledger: block LEDGER only, pass --blocks LEDGER (and --seeds from 61,73,89) ($ledger_protocol)"
            ledger_arm=(--evidence-ledger "$ledger") ;;
        *) die "--evidence-ledger must be none, off or on, not $ledger" ;;
    esac
    # An A5 arm (protocols/EXPA_A5_REVIEW.md): none (or no option) adds nothing; neutral and ledger are registered only
    # with --final-turn-policy deadline, no evidence ledger, the default budget and model profile, for block LEDGER.
    local review_arm=()
    case "$review" in
        ""|none) ;;
        neutral|ledger)
            [ "$final_turn" = deadline ] && { [ -z "$ledger" ] || [ "$ledger" = none ]; } && [ -z "$max_calls$context_budget" ] \
                && [ "$MODEL_PROFILE" = "$MODEL_PROFILE_DEFAULT" ] \
                || die "--submit-review $review: registered only with --final-turn-policy deadline, no --evidence-ledger, the default budget (8 responses, 16384 tokens) and model profile $MODEL_PROFILE_DEFAULT (protocols/EXPA_A5_REVIEW.md)"
            [ "$blocks" = LEDGER ] && { [ -z "$seeds" ] || match '^(61|73|89)(,(61|73|89))*$' "$seeds"; } \
                || die "--submit-review $review: block LEDGER only, pass --blocks LEDGER (and --seeds from 61,73,89) (protocols/EXPA_A5_REVIEW.md)"
            review_arm=(--submit-review "$review") ;;
        *) die "--submit-review must be none, neutral or ledger, not $review" ;;
    esac
    # A7 (protocols/EXPA_A7_IMPACT.md): default (or no option) adds nothing; impact is registered only with the default
    # budget and model profile, no final-turn notice, evidence ledger or submit review, for the A7 selection: block P
    # (none x C0, omit_node x C1) and S-VC (vcomplete x C1) at seed 11. run.json freezes the framing.
    local framing_arm=() a7_protocol=protocols/EXPA_A7_IMPACT.md
    case "$framing" in
        ""|default) ;;
        impact)
            [ -z "$max_calls$context_budget" ] && [ "$MODEL_PROFILE" = "$MODEL_PROFILE_DEFAULT" ] \
                && { [ -z "$final_turn" ] || [ "$final_turn" = none ]; } && { [ -z "$ledger" ] || [ "$ledger" = none ]; } \
                && { [ -z "$review" ] || [ "$review" = none ]; } \
                || die "--task-framing impact: registered only with the default budget (8 responses, 16384 tokens), model profile $MODEL_PROFILE_DEFAULT and no --final-turn-policy, --evidence-ledger or --submit-review ($a7_protocol)"
            [ "$blocks" = P,S-VC ] && [ "$seeds" = 11 ] && [ "$conditions" = C0,C1 ] \
                || die "--task-framing impact: the A7 selection only, pass --blocks P,S-VC --seeds 11 --conditions C0,C1 ($a7_protocol)"
            framing_arm=(--task-framing impact) ;;
        *) die "--task-framing must be default or impact, not $framing" ;;
    esac
    local stamp
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    # Created here as thilak.arjun (0600): systemd would create append: targets as root, unreadable to us.
    : > "$ROOT/logs/runner-$stamp.out"
    : > "$ROOT/logs/runner-$stamp.err"
    info "starting $unit: ${#endpoints[@]} endpoint args, concurrency $concurrency per endpoint, ${selection[*]:-all cells}"
    info "model $MODEL_PROFILE ($MODEL_ALIAS); budgets ${budget[*]:-prereg defaults (8 responses, 16384 tokens)}"
    [ "${#policy[@]}" -eq 0 ] || info "final-turn policy $final_turn ($protocol)"
    if [ -n "$a6" ]; then
        info "A6: evidence ledger $ledger at 16 responses and 32768 tokens ($a6_protocol; ledger rules protocols/EXPA_A3_LEDGER.md)"
    else
        [ "${#ledger_arm[@]}" -eq 0 ] || info "evidence ledger $ledger (protocols/EXPA_A3_LEDGER.md)"
    fi
    [ "${#review_arm[@]}" -eq 0 ] || info "submit review $review (protocols/EXPA_A5_REVIEW.md)"
    [ "${#framing_arm[@]}" -eq 0 ] || info "task framing impact ($a7_protocol)"
    sudo -n systemd-run --unit="$unit" --description="Absent Edges Exp A runner $RUN" --collect \
        --property=Type=exec --property=User="$EXPECTED_USER" --property=Group="$EXPECTED_USER" \
        --property=WorkingDirectory="$ROOT" --property=KillMode=mixed --property=KillSignal=SIGTERM \
        --property=TimeoutStopSec=1800 --property=Restart=no --property=RuntimeMaxSec=432000 \
        --property=NoNewPrivileges=yes --property=ProtectSystem=strict --property=ProtectHome=yes \
        --property="InaccessiblePaths=$INACCESSIBLE" --property=UMask=0077 --property="$SHM_READ_ONLY" \
        --property="ReadWritePaths=$ROOT/run $ROOT/scratch $ROOT/ledger $ROOT/logs" \
        --property=StandardOutput="append:$ROOT/logs/runner-$stamp.out" \
        --property=StandardError="append:$ROOT/logs/runner-$stamp.err" \
        --setenv=PATH=/usr/bin:/bin --setenv=LANG=C.UTF-8 --setenv=PYTHONDONTWRITEBYTECODE=1 \
        --setenv=HOME="$ROOT/scratch/home" --setenv=TMPDIR="$ROOT/scratch/tmp" -- \
        "$PY" -I -B "$REL/scripts/expA_run.py" --dispatch "$ROOT/inputs/dispatch" \
        --checkout-manifest "$ROOT/inputs/checkouts/manifest.json" --repos-root "$ROOT/inputs/checkouts/repos" \
        --run-dir "$ROOT/run" --run-id "$RUN_ID" --ledger "$ROOT/ledger/ledger.sqlite3" --scratch "$ROOT/scratch" \
        "${endpoints[@]}" --concurrency "$concurrency" --git-sha "$GIT_SHA" "${selection[@]}" \
        "${model[@]}" "${budget[@]}" "${policy[@]}" "${ledger_arm[@]}" "${review_arm[@]}" "${framing_arm[@]}"
    info "runner logs: $ROOT/logs/runner-$stamp.{out,err}"
}

publish_plan() {
    local script=$REL/scripts/expA_publish.py sha
    sha=$(release_sha scripts/expA_publish.py)
    check_sha "$sha" "release sha256 of scripts/expA_publish.py"
    "$PY" -I -B "$script" plan --run-root "$ROOT" --target-parent "$TARGET_PARENT" --run-id "$RUN_ID" \
        --script "$script" --script-sha256 "$sha" --interval 30 > "$CONTROL/publish.plan.json.tmp"
    mv -f "$CONTROL/publish.plan.json.tmp" "$CONTROL/publish.plan.json"
}

cmd_publisher() {
    require_identity
    local run=""
    while [ $# -gt 0 ]; do case $1 in --run) run=$2; shift 2 ;; *) die "publisher: unknown option $1" ;; esac; done
    load_config "$run"
    verify_release
    local unit=ae-expA-publish-$RUN argv
    ! systemctl is-active --quiet "$unit" || die "$unit is already active"
    [ ! -e "$CONTROL/publish.stop" ] || die "$CONTROL/publish.stop exists; this run's publishing was finished"
    publish_plan
    mapfile -d '' -t argv < <(json_argv "$CONTROL/publish.plan.json" units init argv)
    info "creating or verifying the Core2 run directory $TARGET_PARENT/$RUN_ID (root, hardened, NFS); it refuses a"
    info "target that already holds this run's records while $ROOT/run has none (restore with stage --restore)"
    sudo -n "${argv[@]}"
    # The loop unit is --collect and Restart=on-failure (transient errors); clear a unit left failed before that.
    sudo -n systemctl reset-failed "$unit" 2>/dev/null || true
    mapfile -d '' -t argv < <(json_argv "$CONTROL/publish.plan.json" units loop argv)
    info "starting $unit (publishes records every 30 s until $CONTROL/publish.stop exists or every cell completed)"
    sudo -n "${argv[@]}"
}

cmd_status() {
    require_identity
    local run=""
    while [ $# -gt 0 ]; do case $1 in --run) run=$2; shift 2 ;; *) die "status: unknown option $1" ;; esac; done
    load_config "$run"
    local unit
    for unit in $(each_instance | awk -v r="$RUN" '{print "ae-expA-serve-"r"-"$1}') "ae-expA-runner-$RUN" "ae-expA-publish-$RUN"; do
        printf '%-48s %s\n' "$unit" "$(systemctl show "$unit" -p ActiveState,SubState,Result --value --no-pager | tr '\n' ' ')"
    done
    gpu_report
    if [ -f "$ROOT/run/run.json" ]; then
        "$PY" -I -B "$REL/scripts/expA_run.py" status --run-dir "$ROOT/run" | head -80 || true
    fi
    journalctl -u "ae-expA-publish-$RUN" -n 3 -o cat --no-pager 2>/dev/null || true
    df -h /dev/shm | tail -1
}

cmd_drain() {
    require_identity
    local run=""
    while [ $# -gt 0 ]; do case $1 in --run) run=$2; shift 2 ;; *) die "drain: unknown option $1" ;; esac; done
    load_config "$run"
    sudo -n systemctl kill --kill-who=main --signal=SIGTERM "ae-expA-runner-$RUN"
    info "sent SIGTERM to the runner's main process: in-flight episodes finish, no new cells start"
}

cmd_stop_runner() {
    require_identity
    local run=""
    while [ $# -gt 0 ]; do case $1 in --run) run=$2; shift 2 ;; *) die "stop-runner: unknown option $1" ;; esac; done
    load_config "$run"
    sudo -n systemctl stop "ae-expA-runner-$RUN" || true
    info "runner stopped: $(systemctl show "ae-expA-runner-$RUN" -p ActiveState --value --no-pager 2>/dev/null)"
}

cmd_stop_serve() {
    require_identity
    local run="" only="" name gpu port
    while [ $# -gt 0 ]; do
        case $1 in --run) run=$2 ;; --instance) only="$only $2" ;; *) die "stop-serve: unknown option $1" ;; esac
        shift 2
    done
    load_config "$run"
    while read -r name gpu port; do
        if [ -n "$only" ]; then case " $only " in *" $name "*) ;; *) continue ;; esac; fi
        sudo -n systemctl stop "ae-expA-serve-$RUN-$name" || true
        info "stopped ae-expA-serve-$RUN-$name"
    done < <(each_instance)
    gpu_report
}

cmd_final_publish() {
    require_identity
    local run="" waited=0 argv status=0 name gpu port
    while [ $# -gt 0 ]; do case $1 in --run) run=$2; shift 2 ;; *) die "final-publish: unknown option $1" ;; esac; done
    load_config "$run"
    verify_release
    ! systemctl is-active --quiet "ae-expA-runner-$RUN" || die "the runner is still active; drain or stop it first"
    # The final pass publishes the vLLM logs: they must not grow afterwards (stop-serve appends shutdown lines).
    while read -r name gpu port; do
        ! systemctl is-active --quiet "ae-expA-serve-$RUN-$name" \
            || die "ae-expA-serve-$RUN-$name is still active; stop-serve first (its log would change after publishing)"
    done < <(each_instance)
    # Close attempts whose runner was killed (owner verified gone): otherwise their events.jsonl is never
    # closed, never published, and the final pass reports them waiting (exit 5).
    info "reconcile: closing abandoned attempts in $ROOT/run (hardened one-shot unit as $EXPECTED_USER)"
    sudo -n systemd-run --unit="ae-expA-reconcile-$RUN" --description="Absent Edges Exp A reconcile $RUN" \
        --wait --pipe --collect --property=Type=exec --property=User="$EXPECTED_USER" \
        --property=Group="$EXPECTED_USER" --property=NoNewPrivileges=yes --property=ProtectSystem=strict \
        --property=ProtectHome=yes --property="InaccessiblePaths=$INACCESSIBLE" --property=UMask=0077 \
        --property="$SHM_READ_ONLY" --property="ReadWritePaths=$ROOT/run $ROOT/ledger" \
        --setenv=PATH=/usr/bin:/bin --setenv=LANG=C.UTF-8 --setenv=PYTHONDONTWRITEBYTECODE=1 -- \
        "$PY" -I -B "$REL/scripts/expA_run.py" reconcile --run-dir "$ROOT/run" \
        --ledger "$ROOT/ledger/ledger.sqlite3" --run-id "$RUN_ID" || die "reconcile failed"
    : > "$CONTROL/publish.stop"
    while systemctl is-active --quiet "ae-expA-publish-$RUN"; do
        [ "$waited" -lt 1800 ] || die "the publish loop did not exit within 30 minutes"
        sleep 5; waited=$((waited + 5))
    done
    publish_plan
    mapfile -d '' -t argv < <(json_argv "$CONTROL/publish.plan.json" units final argv)
    info "final hardened publish pass (records, receipts and logs)"
    sudo -n "${argv[@]}" || status=$?
    if [ "$status" -eq 0 ]; then
        date -u +%Y-%m-%dT%H:%M:%SZ > "$CONTROL/final-publish.ok"
        info "final publish complete: no conflicts, nothing waiting"
    else
        die "final publish exited $status (4 = conflicts under publish/conflicts on Core2; 5 = records still waiting,"\
" listed in publish/finished-*.json; the run root is kept)"
    fi
}

cmd_cleanup() {
    require_identity
    local run="" confirm=0
    while [ $# -gt 0 ]; do
        case $1 in --run) run=$2; shift 2 ;; --confirm) confirm=1; shift ;; *) die "cleanup: unknown option $1" ;; esac
    done
    load_config "$run"
    [ "$confirm" -eq 1 ] || die "cleanup removes $ROOT; pass --confirm"
    [ -f "$CONTROL/final-publish.ok" ] || die "no clean final publish recorded; run final-publish first"
    local units
    units=$(systemctl list-units --all --plain --no-legend "ae-expA-*-$RUN*" | awk '$3=="active"{print $1}')
    [ -z "$units" ] || die "units still active: $units"
    chmod -R u+w "$ROOT/scratch" 2>/dev/null || true
    rm -rf --one-file-system "$ROOT"
    info "removed $ROOT"
}

cmd_show_config() {
    require_identity
    local run=""
    while [ $# -gt 0 ]; do case $1 in --run) run=$2; shift 2 ;; *) die "show-config: unknown option $1" ;; esac; done
    load_config "$run"
    cat "$ROOT/control/launch.env"
}

main() {
    local command=${1:-help}
    shift || true
    case $command in
        gpus) cmd_gpus "$@" ;;
        prepare) cmd_prepare "$@" ;;
        add-instance) cmd_add_instance "$@" ;;
        stage) cmd_stage "$@" ;;
        serve) cmd_serve "$@" ;;
        serve-status) cmd_serve_status "$@" ;;
        runner) cmd_runner "$@" ;;
        publisher) cmd_publisher "$@" ;;
        status) cmd_status "$@" ;;
        drain) cmd_drain "$@" ;;
        stop-runner) cmd_stop_runner "$@" ;;
        stop-serve) cmd_stop_serve "$@" ;;
        final-publish) cmd_final_publish "$@" ;;
        cleanup) cmd_cleanup "$@" ;;
        show-config) cmd_show_config "$@" ;;
        help|-h|--help) printf '%s\n' "commands: gpus prepare add-instance stage serve serve-status runner publisher status drain" \
                            "          stop-runner stop-serve final-publish cleanup show-config (docs/EXPA_RUNBOOK.md)" ;;
        *) die "unknown command $command (see docs/EXPA_RUNBOOK.md)" ;;
    esac
}

main "$@"
