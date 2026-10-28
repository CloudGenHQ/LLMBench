#!/usr/bin/env bash
# ──────────────────────────────────────────────
#  LLM Benchmark Tool – Linux launcher
#  Detects a missing Python 3.8+, offers to install
#  it via the system package manager, then runs the tool.
# ──────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# ── small helpers ─────────────────────────────
info() { echo "[LLMBench] $*" >&2; }
warn() { echo "[LLMBench] WARNING: $*" >&2; }
err()  { echo "[LLMBench] ERROR: $*" >&2; }

# Ask a yes/no question; returns 0 for yes, 1 for no.
# In a non-interactive session (no TTY) it defaults to "no".
ask_yes_no() {
    local answer
    if [ ! -t 0 ]; then
        return 1
    fi
    read -r -p "$1 " answer
    case "${answer:-}" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

# Name of the available system package manager (or "unknown").
detect_pkg_manager() {
    if   command -v apt-get      &>/dev/null; then echo "apt"
    elif command -v dnf          &>/dev/null; then echo "dnf"
    elif command -v yum          &>/dev/null; then echo "yum"
    elif command -v pacman       &>/dev/null; then echo "pacman"
    elif command -v zypper       &>/dev/null; then echo "zypper"
    elif command -v apk          &>/dev/null; then echo "apk"
    elif command -v xbps-install &>/dev/null; then echo "xbps"
    elif command -v pkg          &>/dev/null; then echo "pkg"
    else echo "unknown"
    fi
}

# "sudo" when not root and sudo exists, otherwise empty.
sudo_cmd() {
    if [ "$(id -u)" -ne 0 ] && command -v sudo &>/dev/null; then
        echo "sudo"
    fi
}

# Install Python 3 using the detected package manager.
install_python() {
    local pm SUDO
    pm="$(detect_pkg_manager)"
    SUDO="$(sudo_cmd)"

    if [ "$(id -u)" -ne 0 ] && [ -z "$SUDO" ]; then
        err "Not running as root and 'sudo' is not available; cannot install."
        return 1
    fi

    case "$pm" in
        apt)
            info "Installing python3 via apt..."
            $SUDO apt-get update || return 1
            $SUDO apt-get install -y python3 || return 1
            ;;
        dnf)
            info "Installing python3 via dnf..."
            $SUDO dnf install -y python3 || return 1
            ;;
        yum)
            info "Installing python3 via yum..."
            $SUDO yum install -y python3 || return 1
            ;;
        pacman)
            info "Installing python via pacman..."
            $SUDO pacman -S --noconfirm python || return 1
            ;;
        zypper)
            info "Installing python3 via zypper..."
            $SUDO zypper --non-interactive install python3 || return 1
            ;;
        apk)
            info "Installing python3 via apk..."
            $SUDO apk add python3 || return 1
            ;;
        xbps)
            info "Installing python3 via xbps..."
            $SUDO xbps-install -Sy python || return 1
            ;;
        pkg)
            info "Installing python3 via pkg..."
            $SUDO pkg install -y python3 || return 1
            ;;
        *)
            err "No supported package manager found (apt/dnf/yum/pacman/zypper/apk/xbps/pkg)."
            err "Please install Python 3.8+ manually, then re-run this script."
            return 1
            ;;
    esac
}

# True if $1 is a Python interpreter that is 3.8 or newer.
python_ok() {
    local py="$1"
    command -v "$py" &>/dev/null || return 1
    "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' &>/dev/null
}

# ── main ──────────────────────────────────────
if ! python_ok "$PYTHON"; then
    if command -v "$PYTHON" &>/dev/null; then
        warn "'$PYTHON' is present but older than Python 3.8."
    else
        warn "'$PYTHON' was not found on this system."
    fi
    info "This tool requires Python 3.8+."

    if ask_yes_no "Would you like to install Python 3 now? [y/N]"; then
        if install_python; then
            hash -r 2>/dev/null || true
            if python_ok "$PYTHON"; then
                info "Python 3 is ready."
            elif python_ok "python3"; then
                info "System 'python3' is now available; using it."
                PYTHON="python3"
            else
                err "Installation finished but '$PYTHON' still isn't usable."
                err "Open a new shell, or set PYTHON=/path/to/python and re-run."
                exit 1
            fi
        else
            exit 1
        fi
    else
        err "Aborting. Install Python 3.8+ and re-run, or set PYTHON=/path/to/python."
        exit 1
    fi
fi

exec "$PYTHON" "$SCRIPT_DIR/llm_benchmark.py" "$@"
