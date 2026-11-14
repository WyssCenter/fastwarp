#!/bin/sh
# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end test of scripts/install.sh against a fake release folder: install from file:// and
# http://, checksum refusal, PATH lines in the shell profiles, uninstall. Needs cc, curl and python3.
# shellcheck disable=SC2016

set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
installer=$root/scripts/install.sh
work=$(mktemp -d)
server_pid=
cleanup() {
    [ -z "$server_pid" ] || kill "$server_pid" 2>/dev/null || true
    rm -rf "$work"
}
trap cleanup EXIT

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

pass() {
    printf 'ok: %s\n' "$*"
}

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) target=x86_64-unknown-linux-gnu ;;
    Linux-aarch64) target=aarch64-unknown-linux-gnu ;;
    Darwin-arm64) target=aarch64-apple-darwin ;;
    Darwin-x86_64) target=x86_64-apple-darwin ;;
    *) fail "no release target for $(uname -s) $(uname -m)" ;;
esac
for tool in cc curl python3; do
    command -v "$tool" >/dev/null || fail "$tool is needed for this test"
done

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{print $1}'
}

pkg=$work/pkg/fastwarp-$target
mkdir -p "$pkg" "$work/good" "$work/bad"
printf '#include <stdio.h>\nint main(void) { puts("fastwarp 0.0.0-test"); return 0; }\n' >"$work/dummy.c"
cc -o "$pkg/fastwarp" "$work/dummy.c"
cp "$root/LICENSE" "$root/NOTICE" "$pkg/"
archive=fastwarp-$target.tar.gz
tar -C "$work/pkg" -czf "$work/good/$archive" "fastwarp-$target"
printf '%s  %s\n' "$(sha256 "$work/good/$archive")" "$archive" >"$work/good/$archive.sha256"
cp "$work/good/$archive" "$work/bad/"
printf '%064d  %s\n' 0 "$archive" >"$work/bad/$archive.sha256"

home=$work/home
mkdir -p "$home"
base_path=/usr/bin:/bin:/usr/sbin:/sbin
dl=file://$work/good
test_path=$base_path
test_shell=/bin/bash
test_env=FASTWARP_TEST_UNUSED=1
script=$installer

run() {
    env -i HOME="$home" PATH="$test_path" SHELL="$test_shell" TMPDIR="$work" \
        FASTWARP_DOWNLOAD_URL="$dl" "$test_env" sh "$script" "$@"
}

markers() {
    if [ -f "$1" ]; then grep -c 'fastwarp-installer$' "$1" || true; else echo 0; fi
}

bin=$home/.local/bin/fastwarp
printf 'alias ll="ls -l"\n' >"$home/.bashrc"

dl=file://$work/bad
if out=$(run --version 0.0.0 2>&1); then fail "a wrong checksum was accepted: $out"; fi
printf '%s' "$out" | grep -q 'checksum mismatch' || fail "no checksum message: $out"
[ ! -e "$bin" ] || fail "the binary was installed despite the wrong checksum"
[ "$(markers "$home/.profile")" = 0 ] || fail "a profile was changed despite the wrong checksum"
pass "wrong checksum refused, nothing installed"

dl=file://$work/good
if out=$(run 2>&1); then fail "the repository copy installed without a version: $out"; fi
printf '%s' "$out" | grep -q 'not from a release' || fail "no missing-version message: $out"
pass "unstamped installer needs a version"

run --version 0.0.0 >"$work/log" 2>&1 || fail "install failed: $(cat "$work/log")"
[ -x "$bin" ] || fail "$bin missing"
[ "$("$bin")" = "fastwarp 0.0.0-test" ] || fail "$bin is not the release binary"
[ "$(markers "$home/.profile")" = 1 ] || fail ".profile has no PATH line"
[ "$(markers "$home/.bashrc")" = 1 ] || fail ".bashrc has no PATH line"
grep -q 'alias ll' "$home/.bashrc" || fail ".bashrc lost its content"
grep -q "$home/.profile" "$work/log" || fail "the changed files were not reported"
found=$(env -i HOME="$home" PATH="$base_path" sh -c '. "$HOME/.profile"; . "$HOME/.profile"; command -v fastwarp; printf "%s\n" "$PATH"')
[ "$(printf '%s\n' "$found" | head -n 1)" = "$bin" ] || fail "sourcing .profile does not put fastwarp on PATH: $found"
[ "$(printf '%s\n' "$found" | tail -n 1 | tr ':' '\n' | grep -cx "$home/.local/bin")" = 1 ] || fail "sourcing .profile twice repeats the folder in PATH"
pass "install from file:// with PATH lines in .profile and .bashrc"

run --version 0.0.0 >/dev/null 2>&1 || fail "reinstall failed"
[ "$(markers "$home/.profile")" = 1 ] && [ "$(markers "$home/.bashrc")" = 1 ] || fail "reinstall duplicated a PATH line"
pass "reinstall is idempotent"

run --uninstall >"$work/log" 2>&1 || fail "uninstall failed: $(cat "$work/log")"
[ ! -e "$bin" ] || fail "uninstall left $bin"
[ ! -d "$home/.local/bin" ] || fail "uninstall left the empty folder"
[ "$(markers "$home/.profile")" = 0 ] && [ "$(markers "$home/.bashrc")" = 0 ] || fail "uninstall left a PATH line"
grep -q 'alias ll' "$home/.bashrc" || fail "uninstall damaged .bashrc"
pass "uninstall removes the binary, the folder and the PATH lines"

python3 -m http.server --bind 127.0.0.1 --directory "$work/good" 18765 >"$work/http.log" 2>&1 &
server_pid=$!
i=0
until curl -fsS -o /dev/null "http://127.0.0.1:18765/$archive.sha256" 2>/dev/null; do
    i=$((i + 1))
    [ "$i" -lt 50 ] || fail "the HTTP server did not start: $(cat "$work/http.log")"
    sleep 0.2
done
custom="$work/custom dir/bin"
dl=http://127.0.0.1:18765
run --version v0.0.0 --dir "$custom" --no-modify-path >"$work/log" 2>&1 || fail "install over HTTP failed: $(cat "$work/log")"
[ -x "$custom/fastwarp" ] || fail "--dir was not used"
[ "$(markers "$home/.profile")" = 0 ] || fail "--no-modify-path changed .profile"
grep -q 'export PATH=' "$work/log" || fail "--no-modify-path printed no PATH instructions"
pass "install over http:// into --dir with --no-modify-path"

touch "$custom/other-program"
test_env=FASTWARP_INSTALL_DIR=$custom
run --version 0.0.0 >/dev/null 2>&1 || fail "install into FASTWARP_INSTALL_DIR failed"
[ "$(markers "$home/.profile")" = 1 ] || fail "no PATH line for FASTWARP_INSTALL_DIR"
found=$(env -i HOME="$home" PATH="$base_path" sh -c '. "$HOME/.profile"; command -v fastwarp')
[ "$found" = "$custom/fastwarp" ] || fail "the PATH line for a folder with a space does not work: $found"
run --uninstall >"$work/log" 2>&1 || fail "uninstall from FASTWARP_INSTALL_DIR failed"
[ ! -e "$custom/fastwarp" ] || fail "uninstall left $custom/fastwarp"
[ -e "$custom/other-program" ] || fail "uninstall removed another program"
[ "$(markers "$home/.profile")" = 1 ] || fail "uninstall removed the PATH line of a folder that holds other programs"
grep -q 'Kept the PATH lines' "$work/log" || fail "no note about the kept PATH line"
rm "$custom/other-program"
run --uninstall >/dev/null 2>&1 || fail "second uninstall failed"
[ "$(markers "$home/.profile")" = 0 ] || fail "uninstall of an empty folder kept the PATH line"
pass "PATH line kept while the folder holds other programs"

test_env=FASTWARP_NO_MODIFY_PATH=1
run --version 0.0.0 --dir "$custom" >/dev/null 2>&1 || fail "install with FASTWARP_NO_MODIFY_PATH failed"
[ "$(markers "$home/.profile")" = 0 ] || fail "FASTWARP_NO_MODIFY_PATH=1 changed .profile"
run --uninstall --dir "$custom" >/dev/null 2>&1 || fail "uninstall failed"
test_env=FASTWARP_TEST_UNUSED=1
pass "FASTWARP_NO_MODIFY_PATH=1"

dl=file://$work/good
test_path=$home/.local/bin:$base_path
run --version 0.0.0 >"$work/log" 2>&1 || fail "install with the folder on PATH failed"
test_path=$base_path
[ "$(markers "$home/.profile")" = 0 ] || fail "a folder already on PATH was added again"
grep -q 'already on PATH' "$work/log" || fail "no note that the folder is on PATH"
run --uninstall >/dev/null 2>&1 || fail "uninstall failed"
pass "folder already on PATH: profiles unchanged"

mkdir -p "$home/.local/bin"
ln -s /bin/true "$bin"
if run --version 0.0.0 >"$work/log" 2>&1; then fail "a symbolic link (uv tool) was overwritten"; fi
grep -q 'not installed by this installer' "$work/log" || fail "no message about the foreign file"
if run --uninstall >/dev/null 2>&1; then fail "uninstall removed a symbolic link"; fi
[ -L "$bin" ] || fail "the symbolic link was removed"
rm "$bin"
printf '#!/bin/sh\necho pip\n' >"$bin"
if run --version 0.0.0 >/dev/null 2>&1; then fail "a script (pip entry point) was overwritten"; fi
rm -rf "$home/.local/bin"
pass "foreign fastwarp (symbolic link, script) left alone"

test_shell=/usr/bin/zsh
run --version 0.0.0 >/dev/null 2>&1 || fail "install for zsh failed"
[ "$(markers "$home/.zshrc")" = 1 ] || fail "no PATH line in .zshrc"
test_shell=/usr/bin/fish
run --version 0.0.0 >/dev/null 2>&1 || fail "install for fish failed"
test_shell=/bin/bash
fish_conf=$home/.config/fish/conf.d/fastwarp-installer.fish
[ "$(markers "$fish_conf")" = 1 ] || fail "no fish conf.d file"
[ "$(markers "$home/.zshrc")" = 1 ] || fail "the second install duplicated the .zshrc line"
run --uninstall >/dev/null 2>&1 || fail "uninstall failed"
[ "$(markers "$home/.zshrc")" = 0 ] || fail "uninstall left the .zshrc line"
[ ! -e "$fish_conf" ] || fail "uninstall left the fish conf.d file"
pass "zsh and fish profiles"

mkdir -p "$work/fake"
printf '#!/bin/sh\necho SunOS\n' >"$work/fake/uname"
chmod +x "$work/fake/uname"
test_path=$work/fake:$base_path
if out=$(run --version 0.0.0 2>&1); then fail "an unsupported system was accepted"; fi
test_path=$base_path
printf '%s' "$out" | grep -q 'unsupported operating system' || fail "no unsupported-system message: $out"
pass "unsupported operating system refused"

if out=$(run --version '0.0.0;rm' 2>&1); then fail "an invalid version was accepted"; fi
printf '%s' "$out" | grep -q 'invalid version' || fail "no invalid-version message: $out"
pass "invalid version refused"

if out=$(run --version 0.0.0 --dir "$work/a:b" 2>&1); then fail "a folder with ':' was accepted"; fi
printf '%s' "$out" | grep -q 'must not contain' || fail "no message about the folder name: $out"
pass "install folder with ':' refused"

for dl in http://192.0.2.1/fastwarp http://localhost:1@192.0.2.1/fastwarp; do
    if out=$(run --version 0.0.0 2>&1); then fail "plain HTTP without a held checksum was accepted: $dl"; fi
    printf '%s' "$out" | grep -q 'refusing to download over plain HTTP' || fail "no plain-HTTP message for $dl: $out"
done
dl=file://$work/good
pass "plain HTTP refused unless the checksum is held in the installer"

good_sum=$(sha256 "$work/good/$archive")
other_sum=$(printf '%064d' 1)
sed -e "s/@FASTWARP_VERSION@/0.0.0/" -e "s/@FASTWARP_SHA256_$target@/$good_sum/" \
    -e "s/@FASTWARP_SHA256_[a-z0-9_-]*@/$other_sum/" "$installer" >"$work/stamped.sh"
grep -q '@FASTWARP_' "$work/stamped.sh" && fail "the stamped installer still holds a placeholder"
script=$work/stamped.sh
dl=file://$work/bad
run >"$work/log" 2>&1 || fail "the stamped installer did not use its own checksum: $(cat "$work/log")"
grep -q 'held in this installer' "$work/log" || fail "the stamped installer did not report its checksum"
[ "$("$bin")" = "fastwarp 0.0.0-test" ] || fail "the stamped installer installed something else"
run --uninstall >/dev/null 2>&1 || fail "uninstall failed"
mkdir -p "$work/tampered" "$work/pkg/fastwarp-$target/extra"
tar -C "$work/pkg" -czf "$work/tampered/$archive" "fastwarp-$target"
printf '%s  %s\n' "$(sha256 "$work/tampered/$archive")" "$archive" >"$work/tampered/$archive.sha256"
dl=file://$work/tampered
if out=$(run 2>&1); then fail "a replaced archive with a matching .sha256 was accepted: $out"; fi
printf '%s' "$out" | grep -q 'checksum mismatch' || fail "no checksum message for the replaced archive: $out"
[ ! -e "$bin" ] || fail "the replaced archive was installed"
script=$installer
run --version 0.0.0 >/dev/null 2>&1 || fail "the repository installer refused the release's own .sha256"
run --uninstall >/dev/null 2>&1 || fail "uninstall failed"
dl=file://$work/good
pass "the stamped installer checks against its own checksum, not the release's .sha256"

run --help | grep -q -- '--no-modify-path' || fail "--help does not list --no-modify-path"
pass "--help"

echo "all installer tests passed"
