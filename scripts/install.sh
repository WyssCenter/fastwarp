#!/bin/sh
# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0
#
# Installs the fastwarp command line from a GitHub release (Linux and macOS). See --help.

set -eu

INSTALLER_VERSION="@FASTWARP_VERSION@"
SHA256_X86_64_LINUX="@FASTWARP_SHA256_x86_64-unknown-linux-gnu@"
SHA256_AARCH64_LINUX="@FASTWARP_SHA256_aarch64-unknown-linux-gnu@"
SHA256_AARCH64_MACOS="@FASTWARP_SHA256_aarch64-apple-darwin@"
SHA256_X86_64_MACOS="@FASTWARP_SHA256_x86_64-apple-darwin@"
REPO_URL="https://github.com/WyssCenter/fastwarp"
MARKER="# fastwarp-installer"
MIN_GLIBC="2.28"
MIN_MACOS="11.0"
NL='
'

usage() {
    cat <<EOF
fastwarp installer ${INSTALLER_VERSION}

Downloads the fastwarp command line from ${REPO_URL}/releases, checks its
SHA-256 and installs it without administrator rights. The installer of a
release holds the SHA-256 of that release's archives and checks the download
against it; for another version it uses the release's .sha256 file.

Usage: fastwarp-installer.sh [options]

Options:
  --version VERSION  release to install, for example 0.2.0
                     (default: ${INSTALLER_VERSION}, the release this script belongs to)
  --dir DIR          install into DIR (default: \$XDG_BIN_HOME, else ~/.local/bin)
  --no-modify-path   do not add DIR to PATH in your shell profiles
  --uninstall        remove fastwarp from DIR and the PATH lines this installer added
  -h, --help         show this help

Environment:
  FASTWARP_VERSION         same as --version
  FASTWARP_INSTALL_DIR     same as --dir
  FASTWARP_NO_MODIFY_PATH  set to 1 for --no-modify-path
  FASTWARP_DOWNLOAD_URL    download from this folder instead of the GitHub
                           release (https:// or file://; http:// only with the
                           checksum held in the installer, or from localhost)

Unless DIR is on PATH already or --no-modify-path is given, the installer adds
DIR to PATH in ~/.profile and in the profile of your shell (~/.bashrc,
~/.bash_profile, ~/.zshrc, fish conf.d). Each line it adds ends in
'${MARKER}'; --uninstall removes them.
EOF
}

say() {
    printf '%s\n' "$*"
}

err() {
    printf 'fastwarp-installer: error: %s\n' "$*" >&2
    exit 1
}

version_ge() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        na = split(a, x, "."); nb = split(b, y, ".");
        n = (na > nb) ? na : nb;
        for (i = 1; i <= n; i++) {
            p = (i <= na) ? x[i] + 0 : 0; q = (i <= nb) ? y[i] + 0 : 0;
            if (p > q) exit 0;
            if (p < q) exit 1;
        }
        exit 0
    }'
}

detect_target() {
    os=$(uname -s)
    arch=$(uname -m)
    case "$os" in
        Linux)
            if ldd --version 2>&1 | grep -qi musl; then
                err "musl-based Linux (for example Alpine) is not supported; the release is built for glibc ${MIN_GLIBC} or later"
            fi
            glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}') || glibc=
            [ -n "$glibc" ] || err "could not find the glibc version (getconf GNU_LIBC_VERSION); the release needs glibc ${MIN_GLIBC} or later"
            version_ge "$glibc" "$MIN_GLIBC" || err "glibc $glibc is too old; the release needs glibc ${MIN_GLIBC} or later"
            case "$arch" in
                x86_64 | amd64) target=x86_64-unknown-linux-gnu ;;
                aarch64 | arm64) target=aarch64-unknown-linux-gnu ;;
                *) err "unsupported CPU architecture on Linux: $arch (supported: x86_64, aarch64)" ;;
            esac
            ;;
        Darwin)
            macos=$(sw_vers -productVersion 2>/dev/null) || macos=
            if [ -n "$macos" ]; then
                version_ge "$macos" "$MIN_MACOS" || err "macOS $macos is too old; the release needs macOS ${MIN_MACOS} or later"
            fi
            if [ "$arch" = x86_64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || true)" = 1 ]; then
                arch=arm64
            fi
            case "$arch" in
                arm64 | aarch64) target=aarch64-apple-darwin ;;
                x86_64) target=x86_64-apple-darwin ;;
                *) err "unsupported CPU architecture on macOS: $arch (supported: arm64, x86_64)" ;;
            esac
            ;;
        MINGW* | MSYS* | CYGWIN* | Windows_NT)
            err "this installer is for Linux and macOS; on Windows use fastwarp-installer.ps1"
            ;;
        *)
            err "unsupported operating system: $os (supported: Linux, macOS)"
            ;;
    esac
}

have() {
    command -v "$1" >/dev/null 2>&1
}

download() {
    url=$1
    out=$2
    case "$url" in
        file://*)
            cp "${url#file://}" "$out"
            ;;
        https://*)
            if have curl; then
                curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL --retry 3 -o "$out" "$url"
            elif have wget; then
                wget --https-only --secure-protocol=TLSv1_2 -q -O "$out" "$url"
            else
                err "neither curl nor wget is installed"
            fi
            ;;
        http://*)
            if have curl; then
                curl -fsSL -o "$out" "$url"
            elif have wget; then
                wget -q -O "$out" "$url"
            else
                err "neither curl nor wget is installed"
            fi
            ;;
        *)
            err "unsupported download URL: $url"
            ;;
    esac
}

sha256_of() {
    if have sha256sum; then
        sha256sum "$1" | awk '{print $1}'
    elif have shasum; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif have openssl; then
        openssl dgst -sha256 -r "$1" | awk '{print $1}'
    else
        err "no SHA-256 tool found (sha256sum, shasum or openssl); refusing to install an unchecked download"
    fi
}

is_foreign() {
    [ -L "$1" ] && return 0
    [ "$(dd if="$1" bs=2 count=1 2>/dev/null)" = "#!" ]
}

path_has() {
    case ":${PATH}:" in
        *":$1:"*) return 0 ;;
        *) return 1 ;;
    esac
}

shell_quote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

profile_files() {
    shell_name=$(basename "${SHELL:-sh}")
    printf '%s\n' "$HOME/.profile"
    if [ "$shell_name" = bash ] || [ -f "$HOME/.bashrc" ]; then
        printf '%s\n' "$HOME/.bashrc"
    fi
    if [ -f "$HOME/.bash_profile" ]; then
        printf '%s\n' "$HOME/.bash_profile"
    fi
    if [ "$shell_name" = zsh ] || [ -f "${ZDOTDIR:-$HOME}/.zshrc" ]; then
        printf '%s\n' "${ZDOTDIR:-$HOME}/.zshrc"
    fi
}

all_profile_files() {
    printf '%s\n' "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" "${ZDOTDIR:-$HOME}/.zshrc"
}

fish_file() {
    printf '%s\n' "${XDG_CONFIG_HOME:-$HOME/.config}/fish/conf.d/fastwarp-installer.fish"
}

wants_fish() {
    [ "$(basename "${SHELL:-sh}")" = fish ] || [ -d "${XDG_CONFIG_HOME:-$HOME/.config}/fish" ]
}

posix_line() {
    q=$(shell_quote "$1")
    # shellcheck disable=SC2016
    printf 'case ":${PATH}:" in *:%s:*) ;; *) export PATH=%s":$PATH" ;; esac %s\n' "$q" "$q" "$MARKER"
}

fish_line() {
    q=$(shell_quote "$1")
    # shellcheck disable=SC2016
    printf 'contains -- %s $PATH; or set -gx PATH %s $PATH %s\n' "$q" "$q" "$MARKER"
}

has_line() {
    [ -f "$1" ] && grep -Fxq -- "$2" "$1"
}

remove_line() {
    has_line "$1" "$2" || return 1
    kept="$1.fastwarp-installer.tmp"
    : >"$kept" || err "could not create $kept; $1 was left unchanged"
    status=0
    grep -Fxv -- "$2" "$1" >"$kept" || status=$?
    if [ "$status" -gt 1 ]; then
        rm -f "$kept"
        err "could not edit $1; it was left unchanged"
    fi
    if ! cat "$kept" >"$1"; then
        err "could not write $1; its content without the PATH line is in $kept"
    fi
    rm -f "$kept"
}

modify_path() {
    dir=$1
    if path_has "$dir"; then
        say "$dir is already on PATH; no shell profile was changed."
        return
    fi
    line=$(posix_line "$dir")
    changed=
    files=$(profile_files)
    old_ifs=$IFS
    IFS=$NL
    for f in $files; do
        has_line "$f" "$line" && continue
        if [ -s "$f" ] && [ -n "$(tail -c 1 "$f")" ]; then
            printf '\n' >>"$f"
        fi
        printf '%s\n' "$line" >>"$f"
        changed="$changed  $f$NL"
    done
    IFS=$old_ifs
    if wants_fish; then
        ff=$(fish_file)
        fl=$(fish_line "$dir")
        if ! has_line "$ff" "$fl"; then
            mkdir -p "$(dirname "$ff")"
            printf '%s\n' "$fl" >"$ff"
            changed="$changed  $ff$NL"
        fi
    fi
    if [ -n "$changed" ]; then
        say "Added $dir to PATH with a line ending in '$MARKER' in:"
        printf '%s' "$changed"
        say "Open a new terminal to use fastwarp, or run: . \"$HOME/.profile\""
        say "To undo: run the installer with --uninstall, or delete those lines."
    else
        say "The PATH line for $dir is already in your shell profiles; open a new terminal to use fastwarp."
    fi
}

print_path_help() {
    dir=$1
    path_has "$dir" && return
    say "$dir is not on PATH, and the shell profiles were left unchanged (--no-modify-path)."
    say "To use fastwarp, add it to PATH, for example:"
    say "  echo 'export PATH=\"$dir:\$PATH\"' >> ~/.bashrc   (bash; use ~/.zshrc for zsh)"
    say "  fish_add_path '$dir'   (fish)"
}

unmodify_path() {
    dir=$1
    line=$(posix_line "$dir")
    removed=
    files=$(all_profile_files)
    old_ifs=$IFS
    IFS=$NL
    for f in $files; do
        if remove_line "$f" "$line"; then
            removed="$removed  $f$NL"
        fi
    done
    IFS=$old_ifs
    ff=$(fish_file)
    if remove_line "$ff" "$(fish_line "$dir")"; then
        [ -s "$ff" ] || rm -f "$ff"
        removed="$removed  $ff$NL"
    fi
    if [ -n "$removed" ]; then
        say "Removed the PATH line for $dir from:"
        printf '%s' "$removed"
    fi
}

path_line_present() {
    dir=$1
    line=$(posix_line "$dir")
    files=$(all_profile_files)
    old_ifs=$IFS
    IFS=$NL
    for f in $files; do
        if has_line "$f" "$line"; then
            IFS=$old_ifs
            return 0
        fi
    done
    IFS=$old_ifs
    has_line "$(fish_file)" "$(fish_line "$dir")"
}

resolve_dir() {
    if [ -n "$opt_dir" ]; then
        dir=$opt_dir
    elif [ -n "${FASTWARP_INSTALL_DIR:-}" ]; then
        dir=$FASTWARP_INSTALL_DIR
    elif [ -n "${XDG_BIN_HOME:-}" ]; then
        dir=$XDG_BIN_HOME
    else
        [ -n "${HOME:-}" ] || err "HOME is not set; give the folder with --dir"
        dir=$HOME/.local/bin
    fi
    case "$dir" in
        /*) ;;
        *) dir=$(pwd)/$dir ;;
    esac
    while [ "${dir%/}" != "$dir" ] && [ "$dir" != / ]; do
        dir=${dir%/}
    done
    case "$dir" in
        *:* | *"$NL"* | *\\*) err "the install folder must not contain ':', '\\' or a line break: $dir" ;;
    esac
}

expected_sha256() {
    [ "$version" = "$INSTALLER_VERSION" ] || return 0
    case "$target" in
        x86_64-unknown-linux-gnu) h=$SHA256_X86_64_LINUX ;;
        aarch64-unknown-linux-gnu) h=$SHA256_AARCH64_LINUX ;;
        aarch64-apple-darwin) h=$SHA256_AARCH64_MACOS ;;
        x86_64-apple-darwin) h=$SHA256_X86_64_MACOS ;;
        *) h= ;;
    esac
    case "$h" in
        *@*) ;;
        *) printf '%s\n' "$h" ;;
    esac
}

do_uninstall() {
    resolve_dir
    bin=$dir/fastwarp
    if [ -e "$bin" ] || [ -L "$bin" ]; then
        if is_foreign "$bin"; then
            err "$bin was not installed by this installer (a symbolic link or a script, for example from 'uv tool install' or pip); remove it with the tool that installed it"
        fi
        rm -f "$bin"
        say "Removed $bin"
    else
        say "No fastwarp in $dir"
    fi
    for staged in "$dir"/.fastwarp.new.*; do
        [ -f "$staged" ] && rm -f "$staged"
    done
    if [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        rmdir "$dir" 2>/dev/null || true
    fi
    if [ ! -d "$dir" ]; then
        unmodify_path "$dir"
    elif path_line_present "$dir"; then
        say "Kept the PATH lines for $dir in your shell profiles, because the folder holds other programs."
        say "Delete the lines ending in '$MARKER' by hand if you no longer need them."
    fi
}

do_install() {
    detect_target
    version=${opt_version:-${FASTWARP_VERSION:-$INSTALLER_VERSION}}
    version=${version#v}
    case "$version" in
        *@*) err "no version given, and this copy of the installer is not from a release; pass --version or set FASTWARP_VERSION" ;;
        *"$NL"*) err "invalid version: $version" ;;
    esac
    printf '%s\n' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.+-]+)?$' || err "invalid version: $version"
    resolve_dir

    archive=fastwarp-$target.tar.gz
    base=${FASTWARP_DOWNLOAD_URL:-$REPO_URL/releases/download/v$version}
    base=${base%/}
    pinned=$(expected_sha256)
    case "$base" in
        http://*@*) local_http=0 ;;
        http://127.0.0.1:* | http://127.0.0.1/* | http://localhost:* | http://localhost/*) local_http=1 ;;
        *) local_http=0 ;;
    esac
    case "$base" in
        http://*)
            if [ "$local_http" = 0 ]; then
                [ -n "$pinned" ] || err "refusing to download over plain HTTP without a checksum held in the installer: $base"
                say "warning: downloading over plain HTTP from $base" >&2
            fi
            ;;
    esac

    tmp=$(mktemp -d 2>/dev/null || mktemp -d -t fastwarp)
    staged=
    trap 'rm -rf "$tmp"; [ -z "$staged" ] || rm -f "$staged"' EXIT
    trap 'rm -rf "$tmp"; [ -z "$staged" ] || rm -f "$staged"; exit 130' INT TERM

    say "Downloading fastwarp $version for $target from $base/$archive"
    download "$base/$archive" "$tmp/$archive" || err "download failed: $base/$archive"
    if [ -n "$pinned" ]; then
        expected=$pinned
        source_of_sum="the checksum held in this installer"
    else
        download "$base/$archive.sha256" "$tmp/$archive.sha256" || err "download failed: $base/$archive.sha256"
        expected=$(awk 'NR == 1 {print tolower($1)}' "$tmp/$archive.sha256")
        source_of_sum="$archive.sha256"
    fi
    printf '%s\n' "$expected" | grep -Eqx '[0-9a-f]{64}' || err "$source_of_sum is not a SHA-256 checksum"
    actual=$(sha256_of "$tmp/$archive" | tr 'A-F' 'a-f')
    if [ "$actual" != "$expected" ]; then
        err "SHA-256 checksum mismatch for $archive (expected $expected from $source_of_sum, got $actual); nothing was installed"
    fi
    say "Checksum OK ($source_of_sum): $actual"

    mkdir "$tmp/x"
    tar -xzf "$tmp/$archive" -C "$tmp/x" || err "could not extract $archive"
    src=$tmp/x/fastwarp-$target/fastwarp
    [ -f "$src" ] || err "$archive does not contain fastwarp-$target/fastwarp"

    mkdir -p "$dir" || err "could not create $dir"
    bin=$dir/fastwarp
    if { [ -e "$bin" ] || [ -L "$bin" ]; } && is_foreign "$bin"; then
        err "$bin exists and was not installed by this installer (a symbolic link or a script, for example from 'uv tool install' or pip); remove it first or choose another folder with --dir"
    fi
    staged=$dir/.fastwarp.new.$$
    cp "$src" "$staged" || err "could not write to $dir"
    chmod 755 "$staged"
    if ! out=$("$staged" --version 2>&1); then
        rm -f "$staged"
        err "the downloaded fastwarp does not run on this system: $out"
    fi
    mv -f "$staged" "$bin" || err "could not replace $bin"
    staged=
    say "Installed $out to $bin"

    if [ "$opt_no_modify_path" = 1 ]; then
        print_path_help "$dir"
    else
        modify_path "$dir"
    fi

    other=$(command -v fastwarp 2>/dev/null || true)
    if [ -n "$other" ] && [ "$other" != "$bin" ]; then
        say "note: another fastwarp comes first on PATH: $other"
    fi
}

opt_version=
opt_dir=
case "${FASTWARP_NO_MODIFY_PATH:-}" in
    "" | 0) opt_no_modify_path=0 ;;
    *) opt_no_modify_path=1 ;;
esac
action=do_install
while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            [ $# -ge 2 ] || err "--version needs a value"
            opt_version=$2
            shift 2
            ;;
        --version=*)
            opt_version=${1#*=}
            shift
            ;;
        --dir)
            [ $# -ge 2 ] || err "--dir needs a value"
            opt_dir=$2
            shift 2
            ;;
        --dir=*)
            opt_dir=${1#*=}
            shift
            ;;
        --no-modify-path)
            opt_no_modify_path=1
            shift
            ;;
        --uninstall)
            action=do_uninstall
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            err "unknown option: $1 (see --help)"
            ;;
    esac
done

"$action"
