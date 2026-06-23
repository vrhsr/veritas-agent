<# 
    Veritas Agent — Local Demo Launcher
    Run this script to start the full system locally with Docker.
    Usage: .\scripts\demo.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║  🔬 Veritas Agent — Local Demo Launcher  ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Check Docker ──────────────────────────────────────────────
Write-Host "[1/5] Checking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "  ✓ $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check Docker daemon is running
try {
    docker info 2>&1 | Out-Null
    Write-Host "  ✓ Docker daemon is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# ── Check .env ────────────────────────────────────────────────
Write-Host "[2/5] Checking environment..." -ForegroundColor Yellow
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$envFile = Join-Path $projectRoot ".env"

if (-not (Test-Path $envFile)) {
    Write-Host "  ✗ .env file not found. Copying from .env.example..." -ForegroundColor Yellow
    Copy-Item (Join-Path $projectRoot ".env.example") $envFile
    Write-Host "  ⚠ Please edit .env and add your OPENAI_API_KEY before running queries." -ForegroundColor Yellow
} else {
    $envContent = Get-Content $envFile -Raw
    if ($envContent -match "OPENAI_API_KEY=sk-") {
        Write-Host "  ✓ OPENAI_API_KEY configured" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ OPENAI_API_KEY may not be set in .env — queries will fail without it." -ForegroundColor Yellow
    }
}

# ── Build & Start ─────────────────────────────────────────────
Write-Host "[3/5] Starting containers (this may take a few minutes on first run)..." -ForegroundColor Yellow
Set-Location $projectRoot
docker-compose up -d --build 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }

# ── Wait for Health ───────────────────────────────────────────
Write-Host "[4/5] Waiting for API to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    Start-Sleep -Seconds 2
    $attempt++
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
        if ($health.status -eq "ok") {
            Write-Host "  ✓ API is ready! ($($attempt * 2)s)" -ForegroundColor Green
            break
        }
    } catch {
        Write-Host "  ... attempt $attempt/$maxAttempts" -ForegroundColor DarkGray
    }
}

if ($attempt -ge $maxAttempts) {
    Write-Host "  ⚠ API took too long to start. Check logs with: docker-compose logs agent-api" -ForegroundColor Yellow
}

# ── Open Browser ──────────────────────────────────────────────
Write-Host "[5/5] Opening demo..." -ForegroundColor Yellow
$demoUrl = "http://localhost:8000"
Start-Process $demoUrl
Write-Host ""
Write-Host "  ┌─────────────────────────────────────────────┐" -ForegroundColor Green
Write-Host "  │  🚀 Demo running at: $demoUrl   │" -ForegroundColor Green
Write-Host "  │  📊 Dashboard:       http://localhost:8501  │" -ForegroundColor Green
Write-Host "  │  📖 API Docs:        $demoUrl/docs  │" -ForegroundColor Green
Write-Host "  │                                             │" -ForegroundColor Green
Write-Host "  │  Stop: docker-compose down                  │" -ForegroundColor Green
Write-Host "  └─────────────────────────────────────────────┘" -ForegroundColor Green
Write-Host ""
