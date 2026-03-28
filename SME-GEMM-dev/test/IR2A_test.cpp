#include "IR.h"
#include "descriptor.h"
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>

namespace fs = std::filesystem;

static void expect_contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos)
    {
        std::cerr << "missing expected text: " << needle << "\n";
        std::cerr << text << "\n";
        std::abort();
    }
}

static void expect_not_contains(const std::string &text, const std::string &needle)
{
    if (text.find(needle) != std::string::npos)
    {
        std::cerr << "unexpected text: " << needle << "\n";
        std::cerr << text << "\n";
        std::abort();
    }
}

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
        throw std::runtime_error("unexpected transpose type in IR2A test");
    }
}

static void write_verify_inputs(const fs::path &verify_dir, IR::Function &func, const std::string &kernel_text)
{
    {
        std::ofstream kernel(verify_dir / "kernel.s");
        kernel << kernel_text;
    }
    {
        std::ofstream shape(verify_dir / "shape.h");
        shape << "constexpr int64_t M = " << func.M << ";\n";
        shape << "constexpr int64_t N = " << func.N << ";\n";
        shape << "#define TRANS_TYPE " << trans_type_macro(func) << "\n";
    }
}

static int run_shell(const std::string &command) { return std::system(command.c_str()); }

static std::string capture_command_output(const std::string &command)
{
    std::string output;
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe)
    {
        throw std::runtime_error("failed to run command: " + command);
    }
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe))
    {
        output += buffer;
    }
    const int rc = pclose(pipe);
    if (rc != 0)
    {
        throw std::runtime_error("command failed: " + command);
    }
    return output;
}

static void run_shell_or_die(const std::string &command)
{
    const int rc = run_shell(command);
    if (rc != 0)
    {
        throw std::runtime_error("command failed: " + command);
    }
}

static bool illegal_instruction_status(int rc)
{
    if (rc == -1)
    {
        return false;
    }
    if (WIFSIGNALED(rc) && WTERMSIG(rc) == SIGILL)
    {
        return true;
    }
    if (WIFEXITED(rc) && WEXITSTATUS(rc) == 132)
    {
        return true;
    }
    return false;
}

static void expect_common_kernel(const std::string &text)
{
    expect_contains(text, "__Z15gemm_kernel_optxPPdS0_S0_iii:");
    expect_contains(text, ".Lbatch_loop:");
    expect_contains(text, ".Lbatch_end:");
    expect_contains(text, ".Lk_loop_0:");
    expect_contains(text, ".Lk_end_0:");
    expect_not_contains(text, "DEBUG");
}

static bool g_runtime_supported = true;

static void
compile_and_verify(const fs::path &verify_dir, IR::Function &func, const std::string &kernel_text, int batch, int k)
{
    write_verify_inputs(verify_dir, func, kernel_text);
    const std::string dir = verify_dir.string();
    run_shell_or_die("cd \"" + dir + "\" && make clean && make");
    if (!g_runtime_supported)
    {
        return;
    }
    const std::string command =
        "cd \"" + dir + "\" && TIMES_OVERRIDE=1 ./verify " + std::to_string(batch) + " " + std::to_string(k);
    const int rc = run_shell(command);
    if (illegal_instruction_status(rc))
    {
        g_runtime_supported = false;
        std::cerr << "[IR2A] SME runtime verification skipped: host CPU does not execute SME instructions\n";
        return;
    }
    if (rc != 0)
    {
        throw std::runtime_error("command failed: " + command);
    }
}

static std::string
build_and_capture_text_section(const fs::path &verify_dir, IR::Function &func, const std::string &kernel_text)
{
    write_verify_inputs(verify_dir, func, kernel_text);
    const std::string dir = verify_dir.string();
    run_shell_or_die("cd \"" + dir + "\" && make clean && make kernel.o");
    return capture_command_output("cd \"" + dir + "\" && otool -s __TEXT __text kernel.o");
}

static void cleanup_verify_inputs(const fs::path &verify_dir)
{
    std::error_code ec;
    fs::remove(verify_dir / "kernel.s", ec);
    fs::remove(verify_dir / "shape.h", ec);
    fs::remove(verify_dir / "kernel.o", ec);
    fs::remove(verify_dir / "verify", ec);
}

static std::string normalize_text_section(const std::string &section)
{
    std::string normalized;
    bool first = true;
    bool skipped_header = false;
    std::size_t start = 0;
    while (start < section.size())
    {
        const std::size_t end = section.find('\n', start);
        const std::string line = section.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (skipped_header)
        {
            if (!first)
            {
                normalized += '\n';
            }
            normalized += line;
            first = false;
        }
        else
        {
            skipped_header = true;
        }
        if (end == std::string::npos)
        {
            break;
        }
        start = end + 1;
    }
    return normalized;
}

static void run_case(const std::string &name, IR::Function func, int batch, int k)
{
    const fs::path verify_dir = locate_verify_dir();
    const IR::LoweredAResult lowered = IR::LowerToA(func);
    const std::string asm_text = lowered.asm_text;
    const std::string bin_text = lowered.inst_text;

    expect_common_kernel(asm_text);
    expect_common_kernel(bin_text);
    expect_contains(asm_text, "ret");
    expect_contains(bin_text, ".inst 0x");

    const std::string asm_section = build_and_capture_text_section(verify_dir, func, asm_text);
    const std::string bin_section = build_and_capture_text_section(verify_dir, func, bin_text);
    if (normalize_text_section(asm_section) != normalize_text_section(bin_section))
    {
        throw std::runtime_error("ASM and binary text sections differ for case: " + name);
    }

    std::cout << "[IR2A] verifying asm mode for " << name << "\n";
    compile_and_verify(verify_dir, func, asm_text, batch, k);

    std::cout << "[IR2A] verifying binary mode for " << name << "\n";
    compile_and_verify(verify_dir, func, bin_text, batch, k);
    cleanup_verify_inputs(verify_dir);
}

static IR::Function build_mopa_case()
{
    TilePrimitiveDescriptor desc;
    desc.op_type = TilePrimitiveDescriptor::OP_TYPE::SME_MOPA;
    desc.dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    desc.trans_type = TilePrimitiveDescriptor::GEMM_TN;

    VectorDescriptor vec_a;
    vec_a.elements.push_back({0, 0, 4, 1});
    vec_a.elements.push_back({1, 0, 4, 1});
    desc.vec_a.push_back(vec_a);

    VectorDescriptor vec_b;
    vec_b.elements.push_back({0, 0, 4, 1});
    vec_b.elements.push_back({1, 0, 4, 1});
    desc.vec_b.push_back(vec_b);

    IR::Function func(4, 4, 2, TilePrimitiveDescriptor::DTYPE_FP64, TilePrimitiveDescriptor::GEMM_TN);
    func.build(desc);
    func.allocate_za();
    return func;
}

static IR::Function build_sve_case()
{
    TilePrimitiveDescriptor desc;
    desc.op_type = TilePrimitiveDescriptor::OP_TYPE::SVE_MLA;
    desc.dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    desc.trans_type = TilePrimitiveDescriptor::GEMM_TN;

    VectorDescriptor vec_a;
    vec_a.elements.push_back({0, 0, 4, 1});
    vec_a.elements.push_back({1, 0, 4, 1});
    desc.vec_a.push_back(vec_a);

    VectorDescriptor vec_b;
    vec_b.elements.push_back({0, 0, 1, 4});
    vec_b.elements.push_back({1, 0, 1, 4});
    desc.vec_b.push_back(vec_b);

    IR::Function func(4, 1, 2, TilePrimitiveDescriptor::DTYPE_FP64, TilePrimitiveDescriptor::GEMM_TN);
    func.build(desc);
    return func;
}

static IR::Function build_5x9_case()
{
    IR::Function func(5, 9, 1, TilePrimitiveDescriptor::DTYPE_FP64, TilePrimitiveDescriptor::GEMM_TN);

    TilePrimitiveDescriptor tile0;
    tile0.op_type = TilePrimitiveDescriptor::OP_TYPE::SME_MOPA;
    tile0.dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    tile0.trans_type = TilePrimitiveDescriptor::GEMM_TN;
    {
        VectorDescriptor vec_a;
        vec_a.elements.push_back({0, 0, 5, 1});
        tile0.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({0, 0, 8, 1});
        tile0.vec_b.push_back(vec_b);
        func.build(tile0);
    }

    TilePrimitiveDescriptor tile1;
    tile1.op_type = TilePrimitiveDescriptor::SME2_MLA;
    tile1.dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    tile1.trans_type = TilePrimitiveDescriptor::GEMM_TN;
    {
        VectorDescriptor vec_a;
        vec_a.elements.push_back({0, 0, 5});
        tile1.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({0, 8, 1, 5});
        tile1.vec_b.push_back(vec_b);
        func.build(tile1);
    }

    func.allocate_za();
    func.kLoopMerge();
    func.rearrange();
    return func;
}

static IR::Function build_5x9_scalar_case()
{
    IR::Function func(5, 9, 1, TilePrimitiveDescriptor::DTYPE_FP64, TilePrimitiveDescriptor::GEMM_TN);

    TilePrimitiveDescriptor tile0;
    tile0.op_type = TilePrimitiveDescriptor::OP_TYPE::SCALAR;
    tile0.dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    tile0.trans_type = TilePrimitiveDescriptor::GEMM_TN;
    {
        VectorDescriptor vec_a;
        vec_a.elements.push_back({0, 0, 5, 1});

        tile0.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({0, 0, 8, 1});
        tile0.vec_b.push_back(vec_b);
        func.build(tile0);
    }

    TilePrimitiveDescriptor tile1;
    tile1.op_type = TilePrimitiveDescriptor::SCALAR;
    tile1.dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    tile1.trans_type = TilePrimitiveDescriptor::GEMM_TN;
    {
        VectorDescriptor vec_a;
        vec_a.elements.push_back({0, 0, 5, 1});

        tile1.vec_a.push_back(vec_a);
        VectorDescriptor vec_b;
        vec_b.elements.push_back({0, 8, 1, 1});
        tile1.vec_b.push_back(vec_b);
        func.build(tile1);
    }

    func.allocate_za();
    func.kLoopMerge();
    return func;
}

static IR::Function build_mopa_nt_case()
{
    IR::Function func(4, 4, 2, TilePrimitiveDescriptor::DTYPE_FP64, TilePrimitiveDescriptor::GEMM_NT);
    for (int b = 0; b < 2; ++b)
    {
        TilePrimitiveDescriptor desc;
        desc.op_type = TilePrimitiveDescriptor::OP_TYPE::SME_MOPA;
        desc.dtype = TilePrimitiveDescriptor::DTYPE_FP64;
        desc.trans_type = TilePrimitiveDescriptor::GEMM_NT;

        VectorDescriptor vec_a;
        vec_a.elements.push_back({b, 0, 4, 1});
        desc.vec_a.push_back(vec_a);

        VectorDescriptor vec_b;
        vec_b.elements.push_back({b, 0, 4, 1});
        desc.vec_b.push_back(vec_b);

        func.build(desc);
    }
    func.allocate_za();
    func.kLoopMerge();
    func.rearrange();
    return func;
}
int main()
{
    run_case("mopa", build_mopa_case(), 4, 8);
    printf("build_mopa_case passed\n");
    run_case("sve", build_sve_case(), 4, 8);
    printf("build_sve_case passed\n");
    run_case("5x9", build_5x9_case(), 2, 8);
    printf("build_5x9_case passed\n");
    run_case("5x9_scalar", build_5x9_scalar_case(), 2, 8);
    printf("build_5x9_scalar_case passed\n");
    run_case("mopa_nt", build_mopa_nt_case(), 4, 8);
    printf("build_mopa_nt_case passed\n");
    std::cout << "ALL IR2A tests passed!!!!!\n";
    return 0;
}
