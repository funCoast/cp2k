#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRDPARTY_DIR="${ROOT_DIR}/thirdparty"
STAGING_DIR="${THIRDPARTY_DIR}/_staging/armpl"
INSTALL_BASE="${THIRDPARTY_DIR}/_install/armpl"

OS="$(uname -s)"
ARCH="$(uname -m)"

AUTO_AGREE=0

for arg in "$@"; do
    if [[ "$arg" == "-y" ]] || [[ "$arg" == "--yes" ]]; then
        AUTO_AGREE=1
    fi
done

log() {
    echo "[install_armpl] $*"
}

die() {
    echo "[install_armpl][ERROR] $*" >&2
    exit 1
}

find_repo_armpl_root() {
    if [[ -n "${ARMPL_DIR:-}" ]] && [[ -d "${ARMPL_DIR}" ]]; then
        if [[ -d "${ARMPL_DIR}/include" && -d "${ARMPL_DIR}/lib" ]]; then
            echo "${ARMPL_DIR}"
            return 0
        fi
    fi

    if [[ -d "${INSTALL_BASE}" ]]; then
        while IFS= read -r dir; do
            if [[ -d "${dir}/include" && -d "${dir}/lib" ]]; then
                echo "${dir}"
                return 0
            fi
        done < <(find "${INSTALL_BASE}" -maxdepth 2 -type d -name 'armpl_*' | sort)
    fi
    return 1
}

mkdir -p "${INSTALL_BASE}"

if EXISTING_ROOT="$(find_repo_armpl_root)"; then
    log "Using repo-local installation: ${EXISTING_ROOT}"
    [[ -f "${EXISTING_ROOT}/armpl_env_vars.sh" ]] && log "Environment script: ${EXISTING_ROOT}/armpl_env_vars.sh"
    exit 0
fi

bash "${ROOT_DIR}/scripts/fetch_armpl.sh"

case "${OS}" in
    Darwin)
        DMG_PATH="$(find "${STAGING_DIR}" -maxdepth 3 -type f -name '*.dmg' -print -quit 2>/dev/null || true)"
        [[ -n "${DMG_PATH}" ]] || die "No ArmPL DMG found under ${STAGING_DIR}"

        MOUNT_POINT="$(mktemp -d /tmp/armpl-install.XXXXXX)"
        cleanup() {
            if mount | grep -q "on ${MOUNT_POINT} "; then
                hdiutil detach "${MOUNT_POINT}" >/dev/null || true
            fi
            rmdir "${MOUNT_POINT}" 2>/dev/null || true
        }
        trap cleanup EXIT

        log "Mounting ${DMG_PATH}"
        hdiutil attach "${DMG_PATH}" -mountpoint "${MOUNT_POINT}" -nobrowse -readonly >/dev/null

        INSTALLER="$(find "${MOUNT_POINT}" -maxdepth 2 -type f -name '*install.sh' -print -quit 2>/dev/null || true)"
        [[ -n "${INSTALLER}" ]] || die "No installer script found inside ${DMG_PATH}"

        if [[ "${AUTO_AGREE}" != 1 ]]; then
            die "Installing ArmPL accepts the Arm license. Re-run with -y after review to install into ${INSTALL_BASE}."
        fi

        log "Installing ArmPL into ${INSTALL_BASE}"
        /bin/zsh "${INSTALLER}" --install_dir="${INSTALL_BASE}" -y
        ;;
    Linux)
        INSTALLER="$(find "${STAGING_DIR}" -maxdepth 4 -type f -name '*install.sh' -print -quit 2>/dev/null || true)"
        if [[ -n "${INSTALLER}" ]]; then
            if [[ "${AUTO_AGREE}" != 1 ]]; then
                die "Installing ArmPL accepts the Arm license. Re-run with -y after review to install into ${INSTALL_BASE}."
            fi
            log "Installing ArmPL into ${INSTALL_BASE}"
            bash "${INSTALLER}" --install_dir="${INSTALL_BASE}" -y
        else
            PACKAGE="$(find "${STAGING_DIR}" -maxdepth 4 \( -name '*.deb' -o -name '*.rpm' \) -type f -print -quit 2>/dev/null || true)"
            [[ -n "${PACKAGE}" ]] || die "No Linux installer or package found under ${STAGING_DIR}"
            die "Found packaged ArmPL payload at ${PACKAGE}. Add package-install handling for your distro, then install into ${INSTALL_BASE}."
        fi
        ;;
    *)
        die "Unsupported platform: ${OS}/${ARCH}"
        ;;
esac

if INSTALLED_ROOT="$(find_repo_armpl_root)"; then
    log "Installed repo-local ArmPL: ${INSTALLED_ROOT}"
    [[ -f "${INSTALLED_ROOT}/armpl_env_vars.sh" ]] && log "Environment script: ${INSTALLED_ROOT}/armpl_env_vars.sh"
    exit 0
fi

die "Installer finished but no repo-local ArmPL tree was found under ${INSTALL_BASE}"
