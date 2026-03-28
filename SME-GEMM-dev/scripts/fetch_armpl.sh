#!/usr/bin/env bash
set -euo pipefail

VERSION="26.01"
BASE_URL="https://developer.arm.com/-/cdn-downloads/permalink/Arm-Performance-Libraries/Version_${VERSION}"

MAC_PKG="arm-performance-libraries_${VERSION}_macOS.tgz"
LINUX_PKG="arm-performance-libraries_${VERSION}_deb_gcc.tar"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRDPARTY_DIR="${ROOT_DIR}/thirdparty"
DOWNLOAD_DIR="${THIRDPARTY_DIR}/_downloads/armpl"
STAGING_DIR="${THIRDPARTY_DIR}/_staging/armpl"

mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${STAGING_DIR}"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "[armpl] Detected OS   : ${OS}"
echo "[armpl] Detected ARCH : ${ARCH}"

find_existing_armpl_root() {
    local candidates=()

    if [[ -n "${ARMPL_DIR:-}" ]]; then
        candidates+=("${ARMPL_DIR}")
    fi

    if [[ -d "${THIRDPARTY_DIR}/_install/armpl" ]]; then
        while IFS= read -r dir; do
            candidates+=("${dir}")
        done < <(find "${THIRDPARTY_DIR}/_install/armpl" -maxdepth 2 -type d -name 'armpl_*' | sort)
    fi

    if [[ -d "${THIRDPARTY_DIR}/armpl-install" ]]; then
        while IFS= read -r dir; do
            candidates+=("${dir}")
        done < <(find "${THIRDPARTY_DIR}/armpl-install" -maxdepth 2 -type d -name 'armpl_*' | sort)
    fi

    if [[ -d "${STAGING_DIR}" ]]; then
        while IFS= read -r dir; do
            candidates+=("${dir}")
        done < <(find "${STAGING_DIR}" -maxdepth 2 -type d -name 'armpl_*' | sort)
    fi

    if [[ "${ARMPL_USE_SYSTEM_INSTALL:-0}" = 1 ]] && [[ -d /opt/arm ]]; then
        while IFS= read -r dir; do
            candidates+=("${dir}")
        done < <(find /opt/arm -maxdepth 1 -type d -name 'armpl_*' | sort)
    fi

    local candidate
    for candidate in "${candidates[@]-}"; do
        [[ -d "${candidate}" ]] || continue
        if [[ -f "${candidate}/armpl_env_vars.sh" && -d "${candidate}/include" && -d "${candidate}/lib" ]]; then
            echo "${candidate}"
            return 0
        fi
        if [[ -d "${candidate}/include" && -d "${candidate}/lib" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

case "${OS}" in
    Darwin)
        PKG_NAME="${MAC_PKG}"
        ;;
    Linux)
        PKG_NAME="${LINUX_PKG}"
        ;;
    *)
        echo "[armpl] Unsupported platform: ${OS}"
        exit 1
        ;;
esac

if EXISTING_ROOT="$(find_existing_armpl_root)"; then
    echo "[armpl] Using existing installation: ${EXISTING_ROOT}"
    if [[ -f "${EXISTING_ROOT}/armpl_env_vars.sh" ]]; then
        echo "[armpl] Environment script: ${EXISTING_ROOT}/armpl_env_vars.sh"
    fi
    exit 0
fi

URL="${BASE_URL}/${PKG_NAME}"
ARCHIVE_PATH="${DOWNLOAD_DIR}/${PKG_NAME}"

echo "[armpl] Selected package: ${PKG_NAME}"
echo "[armpl] Download URL: ${URL}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "[armpl] Downloading archive..."
    if command -v curl >/dev/null 2>&1; then
        curl -L "${URL}" -o "${ARCHIVE_PATH}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${ARCHIVE_PATH}" "${URL}"
    else
        echo "[armpl] Neither curl nor wget is available."
        exit 1
    fi
else
    echo "[armpl] Using cached archive: ${ARCHIVE_PATH}"
fi

echo "[armpl] Extracting..."
rm -rf "${STAGING_DIR:?}/"*

case "${PKG_NAME}" in
    *.tgz|*.tar.gz)
        tar -xzf "${ARCHIVE_PATH}" -C "${STAGING_DIR}"
        ;;
    *.tar)
        tar -xf "${ARCHIVE_PATH}" -C "${STAGING_DIR}"
        ;;
    *)
        echo "[armpl] Unknown archive format: ${PKG_NAME}"
        exit 1
        ;;
esac

echo "[armpl] Extraction completed."
echo "[armpl] Staged under: ${STAGING_DIR}"

STAGED_ROOT="$(find "${STAGING_DIR}" -maxdepth 3 -type f -name 'armpl_env_vars.sh' -print -quit 2>/dev/null || true)"
if [[ -n "${STAGED_ROOT}" ]]; then
    STAGED_ROOT="$(cd "$(dirname "${STAGED_ROOT}")" && pwd)"
    echo "[armpl] Ready-to-use ArmPL tree found: ${STAGED_ROOT}"
    echo "[armpl] Use with: export ARMPL_DIR=${STAGED_ROOT} && source ${STAGED_ROOT}/armpl_env_vars.sh"
    exit 0
fi

if [[ "${OS}" == "Darwin" ]]; then
    DMG_PATH="$(find "${STAGING_DIR}" -maxdepth 3 -type f -name '*.dmg' -print -quit 2>/dev/null || true)"
    if [[ -n "${DMG_PATH}" ]]; then
        echo "[armpl] macOS package staged but not installed yet: ${DMG_PATH}"
        echo "[armpl] Mount the DMG, run the included installer, then source armpl_env_vars.sh from the installed armpl_* directory."
        exit 0
    fi
fi

if [[ "${OS}" == "Linux" ]]; then
    PKG_PATH="$(find "${STAGING_DIR}" -maxdepth 3 \( -name '*.deb' -o -name '*.rpm' -o -name '*install.sh' \) -type f -print -quit 2>/dev/null || true)"
    if [[ -n "${PKG_PATH}" ]]; then
        echo "[armpl] Linux package staged but not installed yet: ${PKG_PATH}"
        echo "[armpl] Install the package, then source armpl_env_vars.sh from the installed armpl_* directory."
        exit 0
    fi
fi

echo "[armpl] No ready-to-use installation found in staged files yet."
