#!/usr/bin/env bash
# One-click launcher (macOS/Linux). Downloads Node.js (if missing) and all dependencies by itself.
set -e
cd "$(dirname "$0")"

# 1. Node.js 20+ : use system Node if new enough, else a private copy in ./.node
need_node=1
if command -v node >/dev/null 2>&1 && [ "$(node -p 'process.versions.node.split(".")[0]')" -ge 20 ]; then need_node=0; fi
if [ "$need_node" = 1 ]; then
  if [ ! -x .node/bin/node ]; then
    case "$(uname -s)" in Darwin) plat=darwin;; *) plat=linux;; esac
    case "$(uname -m)" in arm64|aarch64) arch=arm64;; *) arch=x64;; esac
    base="https://nodejs.org/dist/latest-v22.x"
    file="$(curl -fsSL "$base/SHASUMS256.txt" | grep -oE "node-v[0-9.]+-$plat-$arch\.tar\.gz" | head -1)"
    [ -n "$file" ] || { echo "Could not find a Node.js $plat-$arch build."; exit 1; }
    echo "Node.js 20+ not found - downloading $file (one time)..."
    mkdir -p .node && curl -fsSL "$base/$file" | tar -xz --strip-components=1 -C .node
  fi
  export PATH="$PWD/.node/bin:$PATH"
fi

# 2. Dependencies (first run only)
if [ ! -f node_modules/.installed ]; then
  echo "Installing dependencies (first run only)..."
  npm install --no-audit --no-fund
  echo ok > node_modules/.installed
fi

# 3. Config + launch
[ -f .env.local ] || { cp .env.example .env.local; echo "Created .env.local (add GEMINI_API_KEY for Gemini; empty = offline)."; }
PORT="${PORT:-3000}"
( sleep 7; (command -v open >/dev/null && open "http://localhost:$PORT") || (command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT") || true ) &
PORT="$PORT" exec npm run dev
