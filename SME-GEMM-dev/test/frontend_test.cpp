#include "IR.h"
#include "argparse/argparse.hpp"
#include "block2Tile.h"
#include "descriptor.h"
#include "frontend.h"

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

enum class Backend
{
    IR2C,
    IR2A_ASM,
    IR2A_BIN,
};

struct LoweredKernel
{
    std::string kernel_text;
    std::vector<std::uint32_t> binary_blob;
};

static fs::path locate_verify_dir()
{
    const fs::path file_path(__FILE__);
    const fs::path file_parent = file_path.has_parent_path() ? file_path.parent_path() : fs::path(".");
    const fs::path candidates[] = {
        fs::path("test/verify"),
        fs::path("../test/verify"),
        fs::path("../../test/verify"),
        file_parent / "verify",
    };
    for (const auto &candidate : candidates)
    {
        if (fs::exists(candidate / "verify.cpp") && fs::exists(candidate / "wrapper.cpp"))
        {
            return fs::absolute(candidate);
        }
    }
    throw std::runtime_error("failed to locate test/verify directory");
}

static fs::path create_parallel_verify_workspace(const fs::path &verify_template_dir, const std::string &artifact_tag)
{
    const fs::path test_dir = verify_template_dir.parent_path();
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto tid = static_cast<unsigned long long>(std::hash<std::thread::id>{}(std::this_thread::get_id()));

    const fs::path run_dir =
        test_dir / std::format("verify_run_{}_{}_{}", artifact_tag, static_cast<unsigned long long>(now), tid);
    fs::create_directories(run_dir);

    const fs::path files_to_copy[] = {
        verify_template_dir / "Makefile",
        verify_template_dir / "verify.cpp",
        verify_template_dir / "wrapper.cpp",
    };

    for (const auto &src : files_to_copy)
    {
        const fs::path dst = run_dir / src.filename();
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
    }
    return run_dir;
}

static const char *trans_type_macro(IR::Function &func)
{
    switch (func.getTransType())
    {
    case TilePrimitiveDescriptor::GEMM_NN:
        return "nn";
    case TilePrimitiveDescriptor::GEMM_NT:
        return "nt";
    case TilePrimitiveDescriptor::GEMM_TN:
        return "tn";
    case TilePrimitiveDescriptor::GEMM_TT:
        return "tt";
    default:
        throw std::runtime_error("unsupported transpose type");
    }
}

static const char *dtype_macro(IR::Function &func)
{
    switch (func.getDtype())
    {
    case TilePrimitiveDescriptor::DTYPE_FP64:
        return "double";
    case TilePrimitiveDescriptor::DTYPE_FP32:
        return "float";
    default:
        throw std::runtime_error("unsupported dtype");
    }
}

static Backend parse_backend(const std::string &backend)
{
    if (backend == "ir2c")
    {
        return Backend::IR2C;
    }
    if (backend == "ir2a-asm")
    {
        return Backend::IR2A_ASM;
    }
    if (backend == "ir2a-bin")
    {
        return Backend::IR2A_BIN;
    }
    throw std::runtime_error("unknown backend: " + backend);
}

static std::string backend_name(Backend backend)
{
    switch (backend)
    {
    case Backend::IR2C:
        return "ir2c";
    case Backend::IR2A_ASM:
        return "ir2a-asm";
    case Backend::IR2A_BIN:
        return "ir2a-bin";
    }
    throw std::runtime_error("unknown backend");
}

static void cleanup_verify_inputs(const fs::path &verify_dir)
{
    std::error_code ec;
    fs::remove(verify_dir / "kernel.cpp", ec);
    fs::remove(verify_dir / "kernel.s", ec);
    fs::remove(verify_dir / "shape.h", ec);
    fs::remove(verify_dir / "kernel.o", ec);
    fs::remove(verify_dir / "verify", ec);
    fs::remove(verify_dir / "test.sh", ec);
}

static std::string sanitize_artifact_component(const std::string &value)
{
    std::string sanitized;
    sanitized.reserve(value.size());
    for (unsigned char ch : value)
    {
        sanitized.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
    }
    return sanitized;
}

static std::string artifact_stem(const std::string &strategy,
                                 Backend backend,
                                 const std::string &layout,
                                 const std::string &dtype,
                                 int M,
                                 int N,
                                 int K,
                                 int batch)
{
    return std::format("{}__{}__{}__{}__M{}_N{}_K{}_B{}",
                       sanitize_artifact_component(strategy),
                       sanitize_artifact_component(backend_name(backend)),
                       sanitize_artifact_component(layout),
                       sanitize_artifact_component(dtype),
                       M,
                       N,
                       K,
                       batch);
}

static void write_verify_inputs(const fs::path &verify_dir,
                                const fs::path &artifact_root_dir,
                                IR::Function &func,
                                const LoweredKernel &lowered_kernel,
                                Backend backend,
                                const std::string &artifact_tag)
{
    cleanup_verify_inputs(verify_dir);

    const fs::path kernel_path = backend == Backend::IR2C ? verify_dir / "kernel.cpp" : verify_dir / "kernel.s";
    {
        std::ofstream kernel_file(kernel_path);
        kernel_file << lowered_kernel.kernel_text;
    }
    {
        std::ofstream shape(verify_dir / "shape.h");
        shape << "constexpr int64_t M = " << func.M << ";\n";
        shape << "constexpr int64_t N = " << func.N << ";\n";
        shape << "#define TRANS_TYPE " << trans_type_macro(func) << "\n";
        shape << "#define FPTYPE " << dtype_macro(func) << "\n";
    }

    const fs::path artifact_dir = artifact_root_dir / "artifacts";
    fs::create_directories(artifact_dir);
    const fs::path saved_kernel_path =
        artifact_dir / (artifact_tag + (backend == Backend::IR2C ? ".kernel.cpp" : ".kernel.s"));
    const fs::path saved_shape_path = artifact_dir / (artifact_tag + ".shape.h");
    const fs::path saved_binary_path = artifact_dir / (artifact_tag + ".bin");
    {
        std::ofstream saved_kernel(saved_kernel_path);
        saved_kernel << lowered_kernel.kernel_text;
    }
    {
        std::ofstream saved_shape(saved_shape_path);
        saved_shape << "constexpr int64_t M = " << func.M << ";\n";
        saved_shape << "constexpr int64_t N = " << func.N << ";\n";
        saved_shape << "#define TRANS_TYPE " << trans_type_macro(func) << "\n";
        saved_shape << "#define FPTYPE " << dtype_macro(func) << "\n";
    }
    if (!lowered_kernel.binary_blob.empty())
    {
        std::ofstream saved_binary(saved_binary_path, std::ios::binary);
        saved_binary.write(
            reinterpret_cast<const char *>(lowered_kernel.binary_blob.data()),
            static_cast<std::streamsize>(lowered_kernel.binary_blob.size() * sizeof(lowered_kernel.binary_blob[0])));
    }

    std::cout << "Saved kernel copy: " << saved_kernel_path << "\n";
    std::cout << "Saved shape copy: " << saved_shape_path << "\n";
    if (!lowered_kernel.binary_blob.empty())
    {
        std::cout << "Saved binary copy: " << saved_binary_path << "\n";
    }
}

static int run_shell(const std::string &command) { return std::system(command.c_str()); }

static void run_shell_or_die(const std::string &command)
{
    const int rc = run_shell(command);
    if (rc != 0)
    {
        throw std::runtime_error("command failed: " + command);
    }
}

static void validate_strategy_layout(const std::string &strategy, TilePrimitiveDescriptor::TRANS_TYPE trans_type)
{
    if (trans_type != TilePrimitiveDescriptor::GEMM_NT)
    {
        return;
    }

    if (strategy == "sve" || strategy == "fuse_sve" || strategy == "sme2" || strategy == "fuse_sme2")
    {
        throw std::runtime_error("NT layout is not supported for SVE/SME2 strategies");
    }
}

static bool should_rearrange(const std::string &strategy, TilePrimitiveDescriptor::TRANS_TYPE trans_type)
{
    if (strategy == "scalar" || strategy == "sve" || strategy == "sme2")
    {
        return false;
    }
    if (strategy == "mopa")
    {
        return trans_type == TilePrimitiveDescriptor::GEMM_NT;
    }
    if (strategy == "fuse_mopa" || strategy == "fuse_sve" || strategy == "fuse_sme2")
    {
        return true;
    }
    return false;
}

static std::pair<std::vector<TilePrimitiveDescriptor>, int> build_primitives(Frontend::Frontend &fe,
                                                                             const std::string &strategy)
{
    Frontend::TileGenerator generator;
    if (strategy == "costmodel")
    {
        return generator.build_strategy_costmodel(fe);
    }
    if (strategy == "scalar")
    {
        return generator.build_strategy_scalar(fe);
    }
    if (strategy == "sve")
    {
        return generator.build_strategy_mla(fe, TilePrimitiveDescriptor::SVE_MLA);
    }
    if (strategy == "fuse_sve")
    {
        return generator.build_strategy_fuse_mla(fe, TilePrimitiveDescriptor::SVE_MLA);
    }
    if (strategy == "mopa")
    {
        return generator.build_strategy_mopa(fe);
    }
    if (strategy == "fuse_mopa")
    {
        return generator.build_strategy_fuse_mopa(fe);
    }
    if (strategy == "strategy1")
    {
        return generator.build_strategy1(fe);
    }
    if (strategy == "sme2")
    {
        return generator.build_strategy_mla(fe, TilePrimitiveDescriptor::SME2_MLA);
    }
    if (strategy == "fuse_sme2")
    {
        return generator.build_strategy_fuse_mla(fe, TilePrimitiveDescriptor::SME2_MLA);
    }
    throw std::runtime_error("unknown strategy: " + strategy);
}

static LoweredKernel lower_kernel(IR::Function &func, Backend backend)
{
    switch (backend)
    {
    case Backend::IR2C:
        return {IR::LowerToC(func), {}};
    case Backend::IR2A_ASM:
    case Backend::IR2A_BIN:
        {
            const IR::LoweredAResult lowered = IR::LowerToA(func);
            if (!lowered.binary.empty())
            {
                std::vector<std::uint32_t> copied(lowered.binary.size(), 0);
                const IR::LoweredAResult lowered_with_addr = IR::LowerToA(func, copied.data());
                // if (lowered.asm_text != lowered_with_addr.asm_text || lowered.inst_text != lowered_with_addr.inst_text)
                // {
                //     throw std::runtime_error("LowerToA produced inconsistent text across binary-address lowering");
                // }
                // if (copied != lowered.binary || lowered_with_addr.binary != lowered.binary)
                // {
                //     throw std::runtime_error("LowerToA failed to copy generated binary to the requested address");
                // }
            }

            return {backend == Backend::IR2A_ASM ? lowered.asm_text : lowered.inst_text, lowered.binary};
            // return {lowered.asm_text, lowered.inst_text, lowered.binary};
        }
    }
    throw std::runtime_error("unknown backend");
}

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("SMELT_frontend_test");
    program.add_argument("M").scan<'i', int>();
    program.add_argument("N").scan<'i', int>();
    program.add_argument("K").scan<'i', int>();
    program.add_argument("batch").scan<'i', int>().default_value(8);
    program.add_argument("-l", "--layout").choices("nn", "nt", "tn", "tt").default_value("tn");
    program.add_argument("-t", "--type").choices("fp64", "fp32").default_value("fp64");
    program.add_argument("-s", "--strategy")
        .choices("costmodel", "scalar", "sve", "fuse_sve", "mopa", "strategy1", "fuse_mopa", "sme2", "fuse_sme2")
        .default_value("fuse_sve");
    program.add_argument("-b", "--backend").choices("ir2c", "ir2a-asm", "ir2a-bin").default_value("ir2c");
    program.add_argument("-p", "--parallel").default_value(false).implicit_value(true);

    try
    {
        program.parse_args(argc, argv);
    } catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const int M = program.get<int>("M");
    const int N = program.get<int>("N");
    const int K = program.get<int>("K");
    const int batch = program.get<int>("batch");
    const std::string layout = program.get<std::string>("layout");
    const std::string strategy = program.get<std::string>("strategy");
    const Backend backend = parse_backend(program.get<std::string>("backend"));
    const bool parallel_mode = program.get<bool>("parallel");

    const TilePrimitiveDescriptor::DTYPE dtype = program.get<std::string>("type") == "fp64"
                                                     ? TilePrimitiveDescriptor::DTYPE_FP64
                                                     : TilePrimitiveDescriptor::DTYPE_FP32;

    TilePrimitiveDescriptor::TRANS_TYPE trans_type = TilePrimitiveDescriptor::GEMM_NN;
    if (layout == "tt")
    {
        trans_type = TilePrimitiveDescriptor::GEMM_TT;
    }
    else if (layout == "tn")
    {
        trans_type = TilePrimitiveDescriptor::GEMM_TN;
    }
    else if (layout == "nt")
    {
        trans_type = TilePrimitiveDescriptor::GEMM_NT;
    }

    try
    {
        validate_strategy_layout(strategy, trans_type);

        const auto st = std::chrono::high_resolution_clock::now();
        Frontend::Frontend fe(M, N, K, batch, dtype, trans_type);
        fe.build();

        auto [primitives, batch_per_step] = build_primitives(fe, strategy);
        IR::Function func(M, N, batch_per_step, dtype, trans_type);
        for (auto &tile : primitives)
        {
            func.build(tile);
        }
        func.allocate_za();
        func.kLoopMerge();
        if (should_rearrange(strategy, trans_type))
        {
            func.rearrange();
        }

        const LoweredKernel kernel = lower_kernel(func, backend);
        const auto ed = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> diff = ed - st;
        std::cout << "Backend: " << backend_name(backend) << "\n";
        std::cout << "Time taken: " << diff.count() << " ms\n";

        const fs::path verify_template_dir = locate_verify_dir();
        const std::string artifact_tag =
            artifact_stem(strategy, backend, layout, program.get<std::string>("type"), M, N, K, batch);
        const fs::path verify_dir =
            parallel_mode ? create_parallel_verify_workspace(verify_template_dir, artifact_tag) : verify_template_dir;
        struct WorkspaceGuard
        {
            fs::path dir;
            bool active = false;
            ~WorkspaceGuard()
            {
                if (active)
                {
                    std::error_code ec;
                    fs::remove_all(dir, ec);
                }
            }
        } workspace_guard{verify_dir, parallel_mode};

        if (parallel_mode)
        {
            std::cout << "Parallel workspace: " << verify_dir << "\n";
        }
        write_verify_inputs(verify_dir, verify_template_dir, func, kernel, backend, artifact_tag);

        std::cout << "Building and executing...\n";
        const std::string dir = verify_dir.string();
        fs::create_directories(verify_dir / "artifacts");
        fs::create_directories(verify_template_dir / "artifacts");

        const fs::path local_verify_log = verify_dir / "artifacts" / (artifact_tag + ".verify.log");
        const fs::path saved_verify_log = verify_template_dir / "artifacts" / (artifact_tag + ".verify.log");

        auto persist_verify_log = [&]() {
            if (!parallel_mode)
            {
                return;
            }
            if (!fs::exists(local_verify_log))
            {
                return;
            }
            std::error_code ec;
            fs::copy_file(local_verify_log, saved_verify_log, fs::copy_options::overwrite_existing, ec);
        };

        run_shell_or_die("cd \"" + dir + "\" && make clean");
        run_shell_or_die("cd \"" + dir + "\" && make");
        std::cout << "Saved verify log: " << (parallel_mode ? saved_verify_log : local_verify_log) << "\n";
        if (backend == Backend::IR2A_BIN)
        {
            const fs::path binary_path = verify_dir / "artifacts" / (artifact_tag + ".bin");
            try
            {
                run_shell_or_die(std::format("cd \"{}\" && KERNEL_BIN_PATH=\"{}\" ./verify {} {} > \"{}\" 2>&1",
                                             dir,
                                             binary_path.string(),
                                             batch,
                                             K,
                                             local_verify_log.string()));
            } catch (...)
            {
                persist_verify_log();
                throw;
            }
        }
        else
        {
            try
            {
                run_shell_or_die(
                    std::format("cd \"{}\" && ./verify {} {} > \"{}\" 2>&1", dir, batch, K, local_verify_log.string()));
            } catch (...)
            {
                persist_verify_log();
                throw;
            }
        }
        persist_verify_log();
        return 0;
    } catch (const std::exception &err)
    {
        std::cerr << err.what() << "\n";
        return 1;
    }
}
