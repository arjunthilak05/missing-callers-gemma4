#!/bin/bash
# Build a frozen Exp A release tree on Core2 from a git commit, exactly as releases/a53c49396 was built
# (verified: git archive tar sha256 62576498… and RELEASE_FILES.sha256 238b3c4c… reproduce byte for byte).
# Usage: build_release.sh <full-commit-sha>
set -euo pipefail
SHA=$1
REPO="D:/C_moved/Documents-work/2026-10-17/yaml-google-the-gemma-4-developer/outputs/absent-edges"
SCR="C:/Users/Hp/AppData/Local/Temp/session/C--Users-Hp/8f7e53e4-ce23-4904-aa51-7d9cd5eff1f9/scratchpad"
X=/mnt/kldata/absent-edges-20261018-core2/expA
SHORT=$(git -C "$REPO" rev-parse --short=9 "$SHA")
FULL=$(git -C "$REPO" rev-parse "$SHA")
git -C "$REPO" archive --format=tar "$FULL" absent_edges manifests/expA scripts tests > "$SCR/$SHORT.tar"
TARSHA=$(sha256sum "$SCR/$SHORT.tar" | cut -c1-64)
echo "commit $FULL short $SHORT tar sha256 $TARSHA"
ssh -o BatchMode=yes ai-core-team-2 "test ! -e $X/releases/$SHORT && test ! -e $X/releases/$SHORT.tar" || { echo "refusing: release $SHORT exists"; exit 3; }
scp -q -o BatchMode=yes "$SCR/$SHORT.tar" "ai-core-team-2:$X/releases/$SHORT.tar"
ssh -o BatchMode=yes ai-core-team-2 "set -euo pipefail
cd $X/releases
echo '$TARSHA  $SHORT.tar' | sha256sum -c --quiet -
chmod 0444 $SHORT.tar
mkdir $SHORT && cd $SHORT && tar -xf ../$SHORT.tar
T=\$(mktemp /tmp/rf.XXXXXX); LC_ALL=C find . -type f ! -path ./RELEASE_FILES.sha256 | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum > \$T; mv \$T RELEASE_FILES.sha256   # never list itself (stage refuses that)
find . -type f -exec chmod 0444 {} + ; find . -type d -exec chmod 0555 {} +
echo RELEASE_SUMS_SHA256=\$(sha256sum RELEASE_FILES.sha256 | cut -c1-64)
echo STAGER_SHA256=\$(sha256sum scripts/expA_stage.py | cut -c1-64)
echo RELEASE_DIR=$X/releases/$SHORT
echo GIT_SHA=$FULL"
