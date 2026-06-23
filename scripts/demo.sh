#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Veritas Agent — Local Demo Launcher
# Run this script to start the full system locally with Docker.
# Usage: bash scripts/demo.sh
# ──────────────────────────────────────────────────────────────
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
GRAY='\033[0;90m'
NC='\033[0m'

echo ""
echo -e "${CYAN}  ╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}  ║  🔬 Veritas Agent — Local Demo Launcher  ║${NC}"
echo -e "${CYAN}  ╚══════════════════════════════════════════╝${NC}"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Check Docker ──────────────────────────────────────────────
echo -e "${YELLOW}[1/5] Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}  ✗ Docker not found. Please install Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ $(docker --version)${NC}"

if ! docker info &> /dev/null; then
    echo -e "${RED}  ✗ Docker daemon is not running. Please start Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Docker daemon is running${NC}"

# ── Check .env ────────────────────────────────────────────────
echo -e "${YELLOW}[2/5] Checking environment...${NC}"
if [ ! -f .env ]; then
    echo -e "${YELLOW}  ✗ .env file not found. Copying from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}  ⚠ Please edit .env and add your OPENAI_API_KEY.${NC}"
else
    if grep -q "OPENAI_API_KEY=sk-" .env; then
        echo -e "${GREEN}  ✓ OPENAI_API_KEY configured${NC}"
    else
        echo -e "${YELLOW}  ⚠ OPENAI_API_KEY may not be set in .env${NC}"
    fi
fi

# ── Build & Start ─────────────────────────────────────────────
echo -e "${YELLOW}[3/5] Starting containers...${NC}"
docker-compose up -d --build 2>&1 | while read line; do echo -e "${GRAY}  $line${NC}"; done

# ── Wait for Health ───────────────────────────────────────────
echo -e "${YELLOW}[4/5] Waiting for API to be ready...${NC}"
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ API is ready! ($((ATTEMPT * 2))s)${NC}"
        break
    fi
    echo -e "${GRAY}  ... attempt $ATTEMPT/$MAX_ATTEMPTS${NC}"
done

if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
    echo -e "${YELLOW}  ⚠ API took too long. Check logs: docker-compose logs agent-api${NC}"
fi

# ── Open Browser ──────────────────────────────────────────────
echo -e "${YELLOW}[5/5] Opening demo...${NC}"
DEMO_URL="http://localhost:8000"

# Cross-platform browser open
if command -v xdg-open &> /dev/null; then
    xdg-open "$DEMO_URL"
elif command -v open &> /dev/null; then
    open "$DEMO_URL"
fi

echo ""
echo -e "${GREEN}  ┌─────────────────────────────────────────────┐${NC}"
echo -e "${GREEN}  │  🚀 Demo running at: $DEMO_URL   │${NC}"
echo -e "${GREEN}  │  📊 Dashboard:       http://localhost:8501  │${NC}"
echo -e "${GREEN}  │  📖 API Docs:        $DEMO_URL/docs  │${NC}"
echo -e "${GREEN}  │                                             │${NC}"
echo -e "${GREEN}  │  Stop: docker-compose down                  │${NC}"
echo -e "${GREEN}  └─────────────────────────────────────────────┘${NC}"
echo ""
