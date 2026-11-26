#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

rustup_version="1.29.3"
rustup_sha256="b1c8cb14e30889e7e2999e6b69e62d9e9b9ffc0f4df8180564d306d9680e6c55"
sample_url="https://github.com/WyssCenter/fastwarp/releases/download/sample-data-1"
target_dir="${CARGO_TARGET_DIR:-$root/target}"
web="docs/_static/web"

step() {
    printf '\n== %s (%s s)\n' "$1" "$SECONDS"
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}

if ! command -v rustup >/dev/null 2>&1; then
    if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
        step "rustup $rustup_version"
        if [ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]; then
            echo "rustup is not installed; get it from https://rustup.rs" >&2
            exit 1
        fi
        tmp="$(mktemp -d)"
        curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$tmp/rustup-init" \
            "https://static.rust-lang.org/rustup/archive/$rustup_version/x86_64-unknown-linux-gnu/rustup-init"
        got="$(sha256 "$tmp/rustup-init")"
        if [ "$got" != "$rustup_sha256" ]; then
            rm -rf "$tmp"
            echo "rustup-init: SHA-256 $got, expected $rustup_sha256" >&2
            exit 1
        fi
        chmod +x "$tmp/rustup-init"
        "$tmp/rustup-init" -y --no-modify-path --profile minimal --default-toolchain none
        rm -rf "$tmp"
    fi
    export PATH="$HOME/.cargo/bin:$PATH"
fi

step "Rust toolchain of rust-toolchain.toml with wasm32-unknown-unknown"
toolchain="$(sed -n 's/^channel = "\(.*\)"\r*$/\1/p' rust-toolchain.toml)"
rustup toolchain install "$toolchain" --profile minimal
rustup target add --toolchain "$toolchain" wasm32-unknown-unknown

bindgen_version="$(awk '/^name = "wasm-bindgen"/ { getline; gsub(/version = |"|\r/, ""); print; exit }' Cargo.lock)"
tools="$target_dir/web-tools"
bindgen="$tools/bin/wasm-bindgen"
if command -v wasm-bindgen >/dev/null 2>&1 \
    && wasm-bindgen --version | grep -F "wasm-bindgen $bindgen_version" >/dev/null; then
    bindgen="$(command -v wasm-bindgen)"
elif ! "$bindgen" --version 2>/dev/null | grep -F "wasm-bindgen $bindgen_version" >/dev/null; then
    step "wasm-bindgen-cli $bindgen_version"
    cargo install --locked --root "$tools" --version "$bindgen_version" wasm-bindgen-cli
fi

step "fastwarp-web for wasm32"
cargo build --locked --release --target wasm32-unknown-unknown -p fastwarp-web

step "JavaScript bindings into $web/pkg"
rm -rf "$web/pkg"
"$bindgen" --target web --no-typescript --out-dir "$web/pkg" \
    "$target_dir/wasm32-unknown-unknown/release/fastwarp_web.wasm"

step "sample data into $web/sample"
mkdir -p "$web/sample"
while read -r name sum; do
    out="$web/sample/$name"
    if [ -f "$out" ] && [ "$(sha256 "$out")" = "$sum" ]; then
        continue
    fi
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$out.part" "$sample_url/$name"
    got="$(sha256 "$out.part")"
    if [ "$got" != "$sum" ]; then
        rm -f "$out.part"
        echo "$name: SHA-256 $got, expected $sum" >&2
        exit 1
    fi
    mv "$out.part" "$out"
done <<'EOF'
mni152nlin2009basym-head-1mm.nii.gz 53e2a58ca0360c8ca4a22332ad4e477d5dd3e7f910d5520f0d7ebae41cd54201
sample-t1w-defaced.nii.gz 6100646d4e17eeb67f4a9cdf821aca788d190a2cb337e849c18b24dff037480b
EOF

ls -l "$web/pkg" "$web/sample"
step "done"
