#!/usr/bin/env bash
# Render the FQ operator heatmap and the SVG figures to PNG, headlessly.
#
#   ./render.sh                 # everything: heatmap light+dark, all SVGs
#   ./render.sh --only svg      # just the SVG figures
#   ./render.sh --only heatmap --scheme dark
#
# Output lands in ./renders/.
#
# WHY THE ENV GYMNASTICS: this box has no root, no docker, no system fonts and
# no fontconfig. Chromium is Playwright's bundled build; its twelve missing
# .so deps and a DejaVu font set were unpacked from Ubuntu .debs into $HOME
# with `apt-get download` + `dpkg-deb -x` (no root required). `./render.sh
# --bootstrap` re-does that from scratch on a fresh box.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HOME:=/home/mbelleau}"

VENV="${FQ_RENDER_VENV:-$HOME/venvs/render}"
LIBS="${FQ_RENDER_LIBS:-$HOME/.local/chromium-libs}"
FONTS="${FQ_RENDER_FONTS:-$HOME/.local/share/fonts}"
FCFILE="${FQ_RENDER_FONTCONFIG:-$HOME/.local/etc/fonts/fonts.conf}"

# Ubuntu 22.04 package set that satisfies chrome-headless-shell's ldd.
CHROMIUM_DEPS=(
  libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0
  libxcomposite1 libxdamage1 libxrandr2 libxkbcommon0 libgbm1 libasound2
  libcups2 libdrm2 libxfixes3 libwayland-client0 libwayland-server0
  libxshmfence1 libxext6 libxrender1 libexpat1 libxi6
)
FONT_DEBS=(fonts-dejavu-core fonts-liberation2)

bootstrap() {
  local work; work="$(mktemp -d)"
  echo "[bootstrap] staging in $work"
  mkdir -p "$work/apt/lists/partial" "$work/apt/cache/archives/partial" "$work/apt/empty" "$work/debs"
  cat >"$work/apt/sources.list" <<'EOF'
deb http://archive.ubuntu.com/ubuntu jammy main universe
deb http://archive.ubuntu.com/ubuntu jammy-updates main universe
deb http://security.ubuntu.com/ubuntu jammy-security main universe
EOF
  cat >"$work/apt/apt.conf" <<EOF
Dir::State::Lists "$work/apt/lists";
Dir::Etc::SourceList "$work/apt/sources.list";
Dir::Etc::SourceParts "$work/apt/empty";
Dir::Cache "$work/apt/cache";
Dir::Etc::Parts "$work/apt/empty";
APT::Get::AllowUnauthenticated "true";
EOF
  export APT_CONFIG="$work/apt/apt.conf"
  # apt-get update/download need no root when every Dir:: is redirected.
  apt-get update -qq
  ( cd "$work/debs" && apt-get download -qq "${CHROMIUM_DEPS[@]}" "${FONT_DEBS[@]}" )
  for d in "$work"/debs/*.deb; do dpkg-deb -x "$d" "$work/unpack"; done

  mkdir -p "$LIBS" "$FONTS" "$(dirname "$FCFILE")" "$HOME/.cache/fontconfig"
  find "$work/unpack" -path '*/lib/x86_64-linux-gnu/*' \
       \( -name '*.so' -o -name '*.so.*' \) -exec cp -a {} "$LIBS/" \;
  find "$work/unpack" -name '*.ttf' -exec cp -a {} "$FONTS/" \;

  cat >"$FCFILE" <<EOF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>$FONTS</dir>
  <cachedir>$HOME/.cache/fontconfig</cachedir>
$(for f in sans-serif ui-sans-serif system-ui; do
cat <<INNER
  <match target="pattern"><test qual="any" name="family"><string>$f</string></test>
    <edit name="family" mode="prepend" binding="same"><string>DejaVu Sans</string></edit></match>
INNER
done)
$(for f in monospace ui-monospace; do
cat <<INNER
  <match target="pattern"><test qual="any" name="family"><string>$f</string></test>
    <edit name="family" mode="prepend" binding="same"><string>DejaVu Sans Mono</string></edit></match>
INNER
done)
</fontconfig>
EOF

  if [ ! -x "$VENV/bin/python" ]; then uv venv "$VENV" --python 3.12; fi
  uv pip install --python "$VENV/bin/python" playwright
  "$VENV/bin/playwright" install chromium
  rm -rf "$work"
  echo "[bootstrap] done: libs=$LIBS fonts=$FONTS venv=$VENV"
}

if [ "${1:-}" = "--bootstrap" ]; then bootstrap; shift; fi

for need in "$VENV/bin/python" "$LIBS" "$FCFILE"; do
  if [ ! -e "$need" ]; then
    echo "missing: $need -- run: $0 --bootstrap" >&2; exit 1
  fi
done

export LD_LIBRARY_PATH="$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FONTCONFIG_FILE="$FCFILE"
export FONTCONFIG_PATH="$(dirname "$FCFILE")"

# chrome-headless-shell whines about a missing dbus and bluez on every start.
# Neither matters for rasterising; drop the noise, keep everything else.
exec "$VENV/bin/python" "$HERE/render_heatmap.py" "$@" \
  2> >(grep -vE 'dbus/bus\.cc|bluez_dbus_manager|Floss manager|Fontconfig error' >&2)
