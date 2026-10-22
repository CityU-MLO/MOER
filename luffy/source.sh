# Source this file from Bash or Zsh; override toolchain roots as needed.
if [ -n "${BASH_VERSION:-}" ]; then
    _luffy_source_file="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _luffy_source_file="${(%):-%x}"
else
    printf '%s\n' 'Source this file from Bash or Zsh.' >&2
    return 1 2>/dev/null || exit 1
fi
_luffy_source_dir="$(CDPATH= cd -- "$(dirname -- "${_luffy_source_file}")" && pwd)"
export CUDA_HOME="${CUDA_HOME:-${_luffy_source_dir}/toolchain/cuda-12.1}"
export GCC_HOME="${GCC_HOME:-${_luffy_source_dir}/toolchain/gcc-11.2.0}"
export MPFR_HOME="${MPFR_HOME:-${_luffy_source_dir}/toolchain/mpfr-4.1.0}"
export MPC_HOME="${MPC_HOME:-${_luffy_source_dir}/toolchain/mpc-0.8.1}"
export MPFR_LEGACY_HOME="${MPFR_LEGACY_HOME:-${_luffy_source_dir}/toolchain/mpfr-2.4.2}"
export GMP_HOME="${GMP_HOME:-${_luffy_source_dir}/toolchain/gmp-4.3.2}"

for _luffy_bin_dir in "${CUDA_HOME}/bin" "${GCC_HOME}/bin"; do
    if [ -d "${_luffy_bin_dir}" ]; then
        export PATH="${_luffy_bin_dir}${PATH:+:${PATH}}"
    fi
done

# Preserve library precedence without adding empty search-path entries.
for _luffy_lib_dir in "${CUDA_HOME}/lib64" "${GCC_HOME}/lib64" \
    "${MPFR_HOME}/lib" "${MPC_HOME}/lib" "${MPFR_LEGACY_HOME}/lib" "${GMP_HOME}/lib"; do
    if [ -d "${_luffy_lib_dir}" ]; then
        export LD_LIBRARY_PATH="${_luffy_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done

unset _luffy_source_file _luffy_source_dir _luffy_bin_dir _luffy_lib_dir
