#include "IR.h" // 假设上面的代码保存在 ir.h 中
#include "descriptor.h"
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <format>
#include <iomanip>

namespace IR
{

class CGenerator
{
private:
    std::stringstream ss;
    int indent_level = 0;
    int var_counter = 0;
    int pred_counter = 0;

    TilePrimitiveDescriptor::TRANS_TYPE trans_type;

    // Map Instruction pointer to C variable name (e.g., "v1")
    std::unordered_map<Instruction *, std::string> var_map;

    // Map Predicate active mask to C variable name (e.g., "pg0")
    std::unordered_map<int64_t, std::string> pred_map;

    bool A_is_transpose()
    {
        return trans_type == TilePrimitiveDescriptor::TRANS_TYPE::GEMM_TN ||
               trans_type == TilePrimitiveDescriptor::GEMM_TT;
    }

    bool B_is_transpose()
    {
        return trans_type == TilePrimitiveDescriptor::TRANS_TYPE::GEMM_NT ||
               trans_type == TilePrimitiveDescriptor::GEMM_TT;
    }

    static std::string offset_A(int column, bool transpose, int k_offset)
    {
        if (!transpose)
        {
            // return "(k +" + std::to_string(k_offset) + " )+ " + std::to_string(column) + " * K";
            return std::format("(k + {}) + {} * K", k_offset, column);
        }
        else
        {
            // return "k * M + " + std::to_string(column) + "";
            return std::format("(k + {}) * M + {}", k_offset, column);
        }
    }

    static std::string offset_B(int column, bool transpose, int k_offset)
    {
        if (transpose)
        {
            // return "k + " + std::to_string(column) + " * K";
            return std::format("(k + {}) + {} * K", k_offset, column);
        }
        else
        {
            // return "k * N + " + std::to_string(column) + "";
            return std::format("(k + {}) * N + {}", k_offset, column);
        }
    }

    std::string indent() { return std::string(indent_level * 4, ' '); }

    const std::string &getVarName(Instruction *inst)
    {
        if (var_map.find(inst) == var_map.end())
        {
            var_map[inst] = "v" + std::to_string(var_counter++);
        }
        return var_map[inst];
    }
    void setVarName(Instruction *inst, const std::string &name) { var_map[inst] = name; }

    const std::string &getPredName(int64_t mask)
    {
        if (pred_map.find(mask) == pred_map.end())
        {
            pred_map[mask] = "pg" + std::to_string(pred_counter++);
        }
        return pred_map[mask];
    }

    std::string getTypeString(Type *type)
    {
        switch (type->type_id)
        {
        case Type::TYPE_SVFP32:
            return "svfloat32_t";
        case Type::TYPE_SVFP64:
            return "svfloat64_t";
        // 简化处理，假设其他sv类型
        case Type::TYPE_FP32:
            return "float";
        case Type::TYPE_FP64:
            return "double";
        default:
            return "/* unsupported type */";
        }
    }

    // 简单的位掩码分析，用于生成 svwhilelt 或 svptrue
    // IR中的Predicate使用 active mask (1LL << (i * shift))
    std::string generatePredInit(int64_t mask, Type::TYPE_ID type_id)
    {
        int shift = (type_id == Type::TYPE_SVFP64) ? 8 : 4;
        int lanes = 64 / shift;
        // 特判全 true
        if (mask == 0x1111111111111111LL || (type_id == Type::TYPE_SVFP64 && mask == 0x0101010101010101LL))
        {
            return type_id == Type::TYPE_SVFP64 ? "svptrue_b64()" : "svptrue_b32()";
        }

        int first = -1;
        int last = -1;

        // 找第一个 / 最后一个置位的 lane
        for (int i = 0; i < lanes; ++i)
        {
            if (mask & (1LL << (i * shift)))
            {
                if (first == -1)
                {
                    first = i;
                }
                last = i;
            }
        }

        // 空 mask（防御）
        if (first == -1)
        {
            // return type_id == Type::TYPE_SVFP64 ? "svfalse()" : "svfalse()";
            return "svpfalse()";
        }

        // 检查是否连续
        for (int i = first; i <= last; ++i)
        {
            if (!(mask & (1LL << (i * shift))))
            {
                // 非连续，保守 fallback
                // 你也可以在这里扩展成 tbl / dup + compare
                return "/* unsupported predicate mask */";
            }
        }

        int M = first;
        int N = last + 1;
        std::string bw = std::to_string(shift == 8 ? 64 : 32);

        // range(0, N)
        if (M == 0)
        {
            return "svwhilelt_b" + bw + "(0, " + std::to_string(N) + ")";
        }

        // range(M, N)
        return "svnot_z("
               "svwhilelt_b" +
               bw + "(0, " + std::to_string(N) +
               "), "
               "svwhilelt_b" +
               bw + "(0, " + std::to_string(M) +
               ")"
               ")";
    }

    void scanPredicates(Function &func)
    {
        // 这里需要访问Function的private成员，实际工程中可能需要friend或getter
        // 假设我们通过某种方式遍历所有指令收集Predicate
        // 由于无法直接访问Function::primitives，我们假设这个逻辑集成在build流程或有遍历接口
        // 为演示，我们在emitFunction时会重新遍历指令
    }

public:
    void generate(Function &func, const std::string &kernel_name)
    {
        trans_type = func.getTransType();
        // Preamble
        ss << "#include <stddef.h>\n";
        ss << "#include <arm_sve.h>\n";
        ss << "#include <arm_sme.h>\n\n";
        std::string dtype = (func.dtype == TilePrimitiveDescriptor::DTYPE_FP32) ? "float" : "double";
        // Function Signature
        ss << "__arm_new(\"za\") \n"; // SME Kernel usually needs new ZA state or preserving
        ss << "void " << kernel_name << "(int64_t batch, " << dtype << "** A, " << dtype << "** B, " << dtype
           << "** C, int, int, int K) __arm_streaming {\n";
        indent_level++;

        // 1. Collect all predicates first (To define them at entry)
        // Accessing primitives is tricky given the IR definition (private).
        // We will assume we can iterate them via a helper or direct friend access.
        // For this implementation, I will cast `func` to a struct with public primitives or assuming friend.
        auto *hack_func = (&func);
        ss << indent() << "constexpr size_t M = " << func.M << ";\n";
        ss << indent() << "constexpr size_t N = " << func.N << ";\n\n";

        ss << indent() << "for (int64_t b = 0; b < batch; b += " << func.batch_per_step << ") { \n";
        indent_level++;

        std::unordered_set<int64_t> seen_masks;

        // Lambda to collect mask from an instruction
        auto collect_mask = [&](Instruction *inst) {
            if (auto *i = IR::dyn_cast<LoadSVEInst>(inst))
            {
                seen_masks.insert(i->predicate.active); // active is private, assuming friend
            }
            else if (auto *i = IR::dyn_cast<StoreSVEInst>(inst))
            {
                seen_masks.insert(i->predicate.active);
            }
            else if (auto *i = IR::dyn_cast<MopaInst>(inst))
            {
                seen_masks.insert(i->pa.active);
                seen_masks.insert(i->pb.active);
            }
            else if (auto *i = IR::dyn_cast<ReadZAInst>(inst))
            {
                seen_masks.insert(i->predicate.active);
            }
            else if (auto *i = IR::dyn_cast<SelInst>(inst))
            {
                seen_masks.insert(i->predicate.active);
            }
        };

        // Note: active is private in Predicate.
        // We need to add `friend class CGenerator;` to Predicate class in IR headers
        // or add a getter `int64_t getActive() const`.
        // Assuming getter `getActive()` exists or we are friends.
        Type::TYPE_ID type_id =
            func.dtype == TilePrimitiveDescriptor::DTYPE_FP32 ? Type::TYPE_SVFP32 : Type::TYPE_SVFP64;
        for (auto *prim : hack_func->primitives)
        {
            // Need access to Primitive::instructions (also private)
            struct HackPrimitive
            {
                std::vector<Instruction *> instructions;
            };
            auto *hack_prim = reinterpret_cast<HackPrimitive *>(prim);

            for (auto *inst : hack_prim->instructions)
            {
                // Ugly casting to access predicates. In real code, add getters.
                // Assuming we can access the `active` field for this logic:
                if (IR::dyn_cast<LoadSVEInst>(inst))
                {
                    seen_masks.insert(((LoadSVEInst *)inst)->predicate.active);
                }
                else if (IR::dyn_cast<StoreSVEInst>(inst))
                {
                    seen_masks.insert(((StoreSVEInst *)inst)->predicate.active);
                }
                else if (IR::dyn_cast<MopaInst>(inst))
                {
                    seen_masks.insert(((MopaInst *)inst)->pa.active);
                    seen_masks.insert(((MopaInst *)inst)->pb.active);
                }
                else if (IR::dyn_cast<ReadZAInst>(inst))
                {
                    seen_masks.insert(((ReadZAInst *)inst)->predicate.active);
                }
                else if (IR::dyn_cast<SelInst>(inst))
                {
                    seen_masks.insert(((SelInst *)inst)->predicate.active);
                }
                else if (IR::dyn_cast<MlaSVEInst>(inst))
                {
                    seen_masks.insert(((MlaSVEInst *)inst)->predicate.active);
                }
                else if (IR::dyn_cast<SVDupMInst>(inst))
                {
                    seen_masks.insert(((SVDupMInst *)inst)->predicate.active);
                }
            }
        }

        // Emit Predicate Definitions
        for (auto mask : seen_masks)
        {
            std::string name = getPredName(mask);
            // Deduce type from mask pattern is hard without context, default to b32 for GEMM FP32
            ss << indent() << "svbool_t " << name << " = " << generatePredInit(mask, type_id) << ";\n";
        }
        ss << "\n";

        // 2. Emit Primitives
        for (auto *prim : func.primitives)
        {
            ss << indent() << "{\n";
            indent_level++;
            for (auto *inst : prim->instructions)
            {
                emitInstruction(inst);
            }
            indent_level--;
            ss << indent() << "}\n";
        }
        indent_level--;
        ss << indent() << "}\n";
        indent_level--;
        ss << "}\n";
    }

    void emitInstruction(Instruction *inst)
    {
        switch (inst->opcode())
        {
        case OPCODE::KLOOP_BEGIN:
            {
                auto *kloop_begin = IR::dyn_cast<KLoopBeginInst>(inst);
                ss << indent() << "svzero_za();\n";
                ss << indent() << "for (int k = 0; k < K; k += " << kloop_begin->step << ") {\n";
                indent_level++;
                break;
            }

        case OPCODE::KLOOP_END:
            indent_level--;
            ss << indent() << "}\n";
            break;

        case OPCODE::LOAD_SVE:
            {
                auto *load = IR::dyn_cast<LoadSVEInst>(inst);
                std::string ptr_base = (load->target == GEMM_A) ? "A" : "B";
                std::string var = getVarName(load);
                // Calculating pointer:
                // Assuming A is [M, K] or [K, M] and B is [K, N].
                // The IR 'column' usually implies the offset in the vector dimension.
                // Inside K loop, we offset by k.
                // Simplified logic: ptr = Base + (k * stride) + offset
                // We use a generic heuristic string here.
                // Note: In real GEMM, A is often transposed (KxM) to load columns contiguously.
                // Let's assume standard pointer arithmetic: Base + k + offset*LeadingDim ??
                // Or simpler: (Base + k * Stride + column).
                // Using logic: "ptr_base + k * <ImplicitStride> + load->column"

                // To make code compilable and generic:
                std::string offset_str;
                if (load->target == GEMM_A)
                {
                    // Assuming A is used as Col Vector, layout might be Transposed to allow contiguous load
                    // Address = A + (column * K) + k
                    // offset_str = std::string("((k + ") + std::to_string(load->k_offset) + ")  * M )+ " +
                    //  std::to_string(load->column);
                    offset_str = offset_A(load->column, A_is_transpose(), load->k_offset);
                }
                else
                {
                    // GEMM_B, usually Row Vector. Layout Row Major.
                    // Address = B + (k * N) + column
                    // Note: Function class has N, but we don't have access here easily without passing it.
                    // We'll generate a symbolic "N"
                    // Correction: The prompt implies M and N are constants in Function.
                    // Better generated code uses standard pointer logic:
                    // offset_str =
                    // "((k + " + std::to_string(load->k_offset) + " )* N + " + std::to_string(load->column) + ")";
                    offset_str = offset_B(load->column, B_is_transpose(), load->k_offset);
                }

                ss << indent() << getTypeString(load->type) << " " << var << " = svld1("
                   << getPredName(load->predicate.active) << ", " << ptr_base << "[b + " << load->batch << "]" << " + "
                   << offset_str << ");\n";
                break;
            }
        case OPCODE::LOAD:
            {
                auto *load = IR::dyn_cast<LoadInst>(inst);
                std::string ptr_base = (load->target == GEMM_A) ? "A" : "B";
                std::string var = getVarName(load);
                // Calculating pointer:
                // Assuming A is [M, K] or [K, M] and B is [K, N].
                // The IR 'column' usually implies the offset in the vector dimension.
                // Inside K loop, we offset by k.
                // Simplified logic: ptr = Base + (k * stride) + offset
                // We use a generic heuristic string here.
                // Note: In real GEMM, A is often transposed (KxM) to load columns contiguously.
                // Let's assume standard pointer arithmetic: Base + k + offset*LeadingDim ??
                // Or simpler: (Base + k * Stride + column).
                // Using logic: "ptr_base + k * <ImplicitStride> + load->column"

                // To make code compilable and generic:
                std::string offset_str;
                if (load->target == GEMM_A)
                {
                    // Assuming A is used as Col Vector, layout might be Transposed to allow contiguous load
                    // Address = A + (column * K) + k
                    // offset_str = "(k  * M )+ " + std::to_string(load->column);
                    offset_str = offset_A(load->column, A_is_transpose(), 0);
                }
                else
                {
                    // GEMM_B, usually Row Vector. Layout Row Major.
                    // Address = B + (k * N) + column
                    // Note: Function class has N, but we don't have access here easily without passing it.
                    // We'll generate a symbolic "N"
                    // Correction: The prompt implies M and N are constants in Function.
                    // Better generated code uses standard pointer logic:
                    // offset_str = "(k * N + " + std::to_string(load->column) + ")";
                    offset_str = offset_B(load->column, B_is_transpose(), 0);
                }

                ss << indent() << "auto" << " " << var << " = " << ptr_base << "[b + " << load->batch << "]" << "["
                   << offset_str << "];\n";
                break;
            }

        case OPCODE::SEL:
            {
                auto *sel = IR::dyn_cast<SelInst>(inst);
                std::string var = getVarName(sel);
                ss << indent() << getTypeString(sel->type) << " " << var << " = svsel("
                   << getPredName(sel->predicate.active) << ", " << getVarName(sel->trueValue) << ", "
                   << getVarName(sel->falseValue) << ");\n";
                break;
            }
        case IR::OPCODE::SVCREATE2:
            {
                auto create2 = IR::dyn_cast<SVCreate2Inst>(inst);
                std::string var = getVarName(create2);
                ss << indent() << "auto " << var << " = svcreate2(" << getVarName(create2->v0) << ", "
                   << getVarName(create2->v1) << ");\n";
                break;
            }
        case IR::OPCODE::SVCREATE4:
            {
                auto create4 = IR::dyn_cast<SVCreate4Inst>(inst);
                std::string var = getVarName(create4);
                ss << indent() << "auto " << var << " = svcreate4(" << getVarName(create4->v0) << ", "
                   << getVarName(create4->v1) << ", " << getVarName(create4->v2) << ", " << getVarName(create4->v3)
                   << ");\n";
                break;
            }
        case IR::OPCODE::MLA_SME:
            {
                auto *mla = IR::dyn_cast<MlaSMEInst>(inst);
                auto za = (mla->a->type->is_fp64()) ? "za64" : "za32";
                auto vg = (isa<SVCreate2Inst>(mla->a)) ? "vg1x2" : "vg1x4";
                int za_id = mla->za->id + mla->lane * za_num(mla->a->type);
                ss << indent() << "svmla_" << za << "_" << vg << "(" << za_id << ", " << getVarName(mla->a) << ", "
                   << getVarName(mla->b) << ");\n";
                break;
            }

        case OPCODE::MOPA:
            {
                auto *mopa = IR::dyn_cast<MopaInst>(inst);
                // svmopa_za32_m(tile, pred_a, pred_b, op_a, op_b)
                std::string za = (mopa->va->type->type_id == Type::TYPE_SVFP64) ? "za64" : "za32";
                // Assuming ZA tile 0
                ss << indent() << "svmopa_" << za << "_m(" << mopa->za->id << ", " << getPredName(mopa->pa.active)
                   << ", " << getPredName(mopa->pb.active) << ", " << getVarName(mopa->va) << ", "
                   << getVarName(mopa->vb) << ");\n";
                break;
            }
        case IR::OPCODE::MLA_SVE:
            {
                auto *sve = IR::dyn_cast<MlaSVEInst>(inst);
                ss << indent() << getVarName(sve->c) << " = svmla_m(" << getPredName(sve->predicate.active) << ", "
                   << getVarName(sve->c) << ", " << getVarName(sve->a) << ", " << getVarName(sve->b) << ");\n";
                setVarName(sve, getVarName(sve->c));
                break;
            }
        case IR::OPCODE::MLA_SCALAR:
            {
                auto *mla = IR::dyn_cast<MlaScalarInst>(inst);
                ss << indent() << getVarName(mla->c) << " += " << getVarName(mla->a) << " * " << getVarName(mla->b)
                   << ";\n";
                setVarName(mla, getVarName(mla->c));
                break;
            }
        case IR::OPCODE::SVDUPM:
            {
                auto *dup = IR::dyn_cast<SVDupMInst>(inst);
                std::string var = getVarName(dup);
                ss << indent() << getTypeString(dup->type) << " " << var << " = svdup_f"
                   << ((dup->type->type_id == Type::TYPE_SVFP64) ? "64" : "32") << "_m(" << getVarName(dup->ori) << ", "
                   << getPredName(dup->predicate.active) << ", " << getVarName(dup->val) << ");\n";
                break;
            }
        case IR::OPCODE::SVDUP:
            {
                auto *dup = IR::dyn_cast<SVDupInst>(inst);
                std::string var = getVarName(dup);
                ss << indent() << getTypeString(dup->type) << " " << var << " = svdup_f"
                   << ((dup->type->type_id == Type::TYPE_SVFP64) ? "64" : "32") << "(" << getVarName(dup->val)
                   << ");\n";
                break;
            }
        case IR::OPCODE::CONSTANT:
            {
                auto constant = IR::dyn_cast<Constant>(inst);
                std::string var = getVarName(constant);
                ss << indent() << "" << getTypeString(constant->type) << " " << var << " = " << constant->val << ";\n";
                break;
            }
        case IR::OPCODE::SVUNDEF:
            {
                auto *undef = IR::dyn_cast<SVUndefInst>(inst);
                std::string var = getVarName(undef);
                std::string intrinsic;
                if (undef->type->type_id == Type::TYPE_SVFP64)
                {
                    intrinsic = "svundef_f64()";
                }
                else if (undef->type->type_id == Type::TYPE_SVFP32)
                {
                    intrinsic = "svundef_f32()";
                }
                else
                {
                    throw std::runtime_error("Unsupported type for SVUndefInst");
                }
                ss << indent() << getTypeString(undef->type) << " " << var << " = " << intrinsic << ";\n";
                break;
            }

        case IR::OPCODE::WRITE_ZA:
            {
                auto write = IR::dyn_cast<WriteZAInst>(inst);
                std::string var = getVarName(write->source);
                std::string dir = (write->direction == WriteZAInst::HORIZONTAL) ? "hor" : "ver";
                std::string za = (write->type->type_id == Type::TYPE_SVFP64) ? "za64" : "za32";
                int za_id = write->za->id;
                // svwrite_hor_za32(za_id, slice_index, pg, var)
                ss << indent() << "svwrite_" << dir << "_" << za << "_m(" << za_id << ", " << write->lane << ", "
                   << getPredName(write->predicate.active) << ", " << var << ");\n";
                break;
            }
        case OPCODE::READ_ZA:
            {
                auto *read = IR::dyn_cast<ReadZAInst>(inst);
                std::string var = getVarName(read);
                std::string dir = (read->direction == ReadZAInst::HORIZONTAL) ? "hor" : "ver";
                std::string za = (read->type->type_id == Type::TYPE_SVFP64) ? "za64" : "za32";
                int za_id = read->za->id;
                // svread_hor_za32(tile, pg, slice_index)
                ss << indent() << getTypeString(read->type) << " " << var << " = svread_" << dir << "_" << za << "_m("
                   << getVarName(read->source) << ", " << getPredName(read->predicate.active) << ", " << za_id << ", "
                   << read->lane << ");\n";
                break;
            }

        case OPCODE::STORE_SVE:
            {
                auto *store = IR::dyn_cast<StoreSVEInst>(inst);
                // svst1(pg, ptr, val)
                // C layout: C + row * N + column
                std::string offset = "(" + std::to_string(store->row) + " * N + " + std::to_string(store->column) + ")";
                ss << indent() << "svst1(" << getPredName(store->predicate.active) << ", C" << "[b + " << store->batch
                   << "] +" << offset << ", " << getVarName(store->val) << ");\n";
                break;
            }
        case IR::OPCODE::STORE:
            {
                auto *store = IR::dyn_cast<StoreInst>(inst);
                std::string offset = "(" + std::to_string(store->row) + " * N + " + std::to_string(store->column) + ")";
                ss << indent() << "C[b + " << store->batch << "][" << offset << "] = " << getVarName(store->val)
                   << ";\n";
                break;
            }

        default:
            ss << indent() << "// Unsupported Opcode: " << inst->opcode() << "\n";
            break;
        }
    }

    std::string getCode() { return ss.str(); }
};

// 暴露给外部调用的函数
std::string LowerToC(Function &func)
{
    // 为了访问private成员，需要在IR类中添加 friend class CGenerator;
    // 或者在编译时修改IR定义。此处假设已解决访问权限问题。
    // 为了完善指针计算，可以将 M 和 N 传递进Generator，这里为简化仅作为参数。
    CGenerator gen;
    // 可以在Generator中存储M和N用于生成正确的Stride
    gen.generate(func, "gemm_kernel_opt");
    return gen.getCode();
}

} // namespace IR