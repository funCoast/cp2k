#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace ARM
{

enum OPCODE
{
    LABEL,
    ZERO,
    MOV_IMM,
    CMP,
    ADD_IMM,
    SUB_IMM,
    ADD_REG,
    LDR,
    LDR_SCALAR,
    STR_SCALAR,
    DUP_GPR,
    FMOV_WX_TO_FP,
    FMOV_FP_TO_WX,
    FMADD_SCALAR,
    LD1D,
    LD1W,
    BGE,
    BLT,
    MOV_T2V,
    MOV_V2T,
    ST1D,
    ST1W,
    EOR,
    PTRUE,
    PFALSE,
    WHILELT,
    SEL,
    FMLA_SVE,
    FMLA_SME,
    FMOPA,
    RET,
};

inline std::string x_or_sp(int reg) { return reg == 31 ? "SP" : std::format("X{}", reg); }

class Instruction
{
public:
    virtual ~Instruction() = default;
    virtual OPCODE opcode() const = 0;
    virtual std::string to_asm() const = 0;
    virtual int to_binary() const = 0;

    virtual void set_p_reg(std::size_t index, int reg)
    {
        (void)index;
        (void)reg;
        throw std::out_of_range("instruction does not expose P register operands");
    }

    virtual void set_wx_reg(std::size_t index, int reg)
    {
        (void)index;
        (void)reg;
        throw std::out_of_range("instruction does not expose W/X register operands");
    }

    virtual void set_z_reg(std::size_t index, int reg)
    {
        (void)index;
        (void)reg;
        throw std::out_of_range("instruction does not expose Z register operands");
    }
};

template <typename T>
bool isa(Instruction *inst)
{
    if (inst == nullptr)
    {
        return false;
    }
    return inst->opcode() == T::StaticOpcode;
}

template <typename T>
T *dyn_cast(Instruction *inst)
{
    static_assert(std::is_base_of_v<Instruction, T>, "T must be derived from ARM::Instruction");
    if (isa<T>(inst))
    {
        return static_cast<T *>(inst);
    }
    return nullptr;
}

class LabelInst : public Instruction
{
    std::string label;
    int binary_address;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::LABEL;
    OPCODE opcode() const override { return StaticOpcode; }
    LabelInst(std::string_view label, int binary_address)
    : label(label)
    , binary_address(binary_address)
    {
    }

    std::string to_asm() const override { return std::format("{}:", label); }

    int to_binary() const override { return 0; }

    const std::string &name() const { return label; }
};

class ZeroInst : public Instruction
{
    int imm8 = 0b11111111u;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::ZERO;
    OPCODE opcode() const override { return StaticOpcode; }
    explicit ZeroInst(int imm)
    : imm8(0b11111111u)
    {
        imm8 = (imm & 0b11111111u);
    }

    std::string to_asm() const override { return "zero { za }"; }

    int to_binary() const override { return static_cast<int>(0xc0080000u | (imm8 << 0)); }
};

class MOVImmInst : public Instruction
{
    int Xd;
    int imm16;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::MOV_IMM;
    OPCODE opcode() const override { return StaticOpcode; }
    MOVImmInst(int xd, int imm)
    : Xd(xd)
    , imm16(imm)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Xd = reg;
            return;
        }
        throw std::out_of_range("MOVImmInst W/X register index out of range");
    }

    std::string to_asm() const override { return std::format("movz X{}, #{}", Xd, imm16); }

    int to_binary() const override { return static_cast<int>((0b11010010100u << 21) | (Xd << 0) | (imm16 << 5)); }
};

class CMPInst : public Instruction
{
    int Xn;
    int Xm;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::CMP;
    OPCODE opcode() const override { return StaticOpcode; }
    CMPInst(int xn, int xm)
    : Xn(xn)
    , Xm(xm)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xn = reg;
            return;
        case 1:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("CMPInst W/X register index out of range");
        }
    }

    std::string to_asm() const override { return std::format("cmp X{}, X{}", Xn, Xm); }

    int to_binary() const override
    {
        return static_cast<int>((0b11101011000u << 21) | (Xn << 5) | (Xm << 16) | (0b11111u << 0));
    }
};

class ADDImmInst : public Instruction
{
    int Xd;
    int Xn;
    int imm12;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::ADD_IMM;
    OPCODE opcode() const override { return StaticOpcode; }
    ADDImmInst(int xd, int xn, int imm)
    : Xd(xd)
    , Xn(xn)
    , imm12(imm)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xd = reg;
            return;
        case 1:
            Xn = reg;
            return;
        default:
            throw std::out_of_range("ADDImmInst W/X register index out of range");
        }
    }

    std::string to_asm() const override { return std::format("add {}, {}, #{}", x_or_sp(Xd), x_or_sp(Xn), imm12); }

    int to_binary() const override
    {
        return static_cast<int>((0b1001000100u << 22) | (Xd << 0) | (Xn << 5) | (imm12 << 10));
    }
};

class SUBImmInst : public Instruction
{
    int Xd;
    int Xn;
    int imm12;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::SUB_IMM;
    OPCODE opcode() const override { return StaticOpcode; }
    SUBImmInst(int xd, int xn, int imm)
    : Xd(xd)
    , Xn(xn)
    , imm12(imm)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xd = reg;
            return;
        case 1:
            Xn = reg;
            return;
        default:
            throw std::out_of_range("SUBImmInst W/X register index out of range");
        }
    }

    std::string to_asm() const override { return std::format("sub {}, {}, #{}", x_or_sp(Xd), x_or_sp(Xn), imm12); }

    int to_binary() const override
    {
        return static_cast<int>((0b1101000100u << 22) | (Xd << 0) | (Xn << 5) | (imm12 << 10));
    }
};

// ADD <Xd>, <Xn>, <Xm>{, <shift>(仅LSL-00) #<amount>}
class ADDRegInst : public Instruction
{
    int Xd;
    int Xn;
    int Xm;
    int imm6;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::ADD_REG;
    OPCODE opcode() const override { return StaticOpcode; }
    ADDRegInst(int xd, int xn, int xm, int imm6)
    : Xd(xd)
    , Xn(xn)
    , Xm(xm)
    , imm6(imm6)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xd = reg;
            return;
        case 1:
            Xn = reg;
            return;
        case 2:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("ADDRegInst W/X register index out of range");
        }
    }

    std::string to_asm() const override
    {
        if (imm6 == 0)
        {
            return std::format("add X{}, X{}, X{}", Xd, Xn, Xm);
        }
        else
        {
            return std::format("add X{}, X{}, X{}, LSL #{}", Xd, Xn, Xm, imm6);
        }
    }

    int to_binary() const override
    {
        return static_cast<int>((0b10001011000u << 21) | (Xd << 0) | (Xn << 5) | (Xm << 16) | (imm6 << 10));
    }
};

// LDR <Xt>, [<Xn>, <Xm>]
class LDRInst : public Instruction
{
    int Xt;
    int Xn;
    int Xm;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::LDR;
    OPCODE opcode() const override { return StaticOpcode; }
    LDRInst(int xt, int xn, int xm)
    : Xt(xt)
    , Xn(xn)
    , Xm(xm)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xt = reg;
            return;
        case 1:
            Xn = reg;
            return;
        case 2:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("LDRInst W/X register index out of range");
        }
    }

    std::string to_asm() const override { return std::format("ldr X{}, [X{}, X{}]", Xt, Xn, Xm); }

    int to_binary() const override
    {
        return static_cast<int>((0b1111100001u << 21) | (Xt << 0) | (Xn << 5) | (Xm << 16) | (0b011010u << 10));
    }
};

class LDRScalarInst : public Instruction
{
    int Rt;
    int Xn;
    int Xm;
    bool is_64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::LDR_SCALAR;
    OPCODE opcode() const override { return StaticOpcode; }
    LDRScalarInst(int rt, int xn, int xm, bool is64)
    : Rt(rt)
    , Xn(xn)
    , Xm(xm)
    , is_64(is64)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Rt = reg;
            return;
        case 1:
            Xn = reg;
            return;
        case 2:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("LDRScalarInst W/X register index out of range");
        }
    }

    std::string to_asm() const override
    {
        return std::format("ldr {}{}, [X{}, X{}, LSL #{}]", is_64 ? "X" : "W", Rt, Xn, Xm, is_64 ? 3 : 2);
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_64 ? 0xf8607800u : 0xb8607800u;
        return static_cast<int>(base | (Xm << 16) | (Xn << 5) | Rt);
    }
};

class STRScalarInst : public Instruction
{
    int Rt;
    int Xn;
    int Xm;
    bool is_64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::STR_SCALAR;
    OPCODE opcode() const override { return StaticOpcode; }
    STRScalarInst(int rt, int xn, int xm, bool is64)
    : Rt(rt)
    , Xn(xn)
    , Xm(xm)
    , is_64(is64)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Rt = reg;
            return;
        case 1:
            Xn = reg;
            return;
        case 2:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("STRScalarInst W/X register index out of range");
        }
    }

    std::string to_asm() const override
    {
        return std::format("str {}{}, [X{}, X{}, LSL #{}]", is_64 ? "X" : "W", Rt, Xn, Xm, is_64 ? 3 : 2);
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_64 ? 0xf8207800u : 0xb8207800u;
        return static_cast<int>(base | (Xm << 16) | (Xn << 5) | Rt);
    }
};

class DUPGPRInst : public Instruction
{
    int Zd;
    int Rn;
    bool is_64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::DUP_GPR;
    OPCODE opcode() const override { return StaticOpcode; }
    DUPGPRInst(int zd, int rn, bool is64)
    : Zd(zd)
    , Rn(rn)
    , is_64(is64)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Rn = reg;
            return;
        }
        throw std::out_of_range("DUPGPRInst W/X register index out of range");
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Zd = reg;
            return;
        }
        throw std::out_of_range("DUPGPRInst Z register index out of range");
    }

    std::string to_asm() const override
    {
        return std::format("dup Z{}.{} , {}{}", Zd, is_64 ? "D" : "S", is_64 ? "X" : "W", Rn);
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_64 ? 0x05e03800u : 0x05a03800u;
        return static_cast<int>(base | (Rn << 5) | Zd);
    }
};

class FMovWXToFPInst : public Instruction
{
    int Vd;
    int Rn;
    bool is_64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::FMOV_WX_TO_FP;
    OPCODE opcode() const override { return StaticOpcode; }
    FMovWXToFPInst(int vd, int rn, bool is64)
    : Vd(vd)
    , Rn(rn)
    , is_64(is64)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Rn = reg;
            return;
        }
        throw std::out_of_range("FMovWXToFPInst W/X register index out of range");
    }

    std::string to_asm() const override
    {
        return std::format("fmov {}{}, {}{}", is_64 ? "D" : "S", Vd, is_64 ? "X" : "W", Rn);
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_64 ? 0x9e670000u : 0x1e260000u;
        return static_cast<int>(base | (Rn << 5) | Vd);
    }
};

class FMovFPToWXInst : public Instruction
{
    int Rd;
    int Vn;
    bool is_64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::FMOV_FP_TO_WX;
    OPCODE opcode() const override { return StaticOpcode; }
    FMovFPToWXInst(int rd, int vn, bool is64)
    : Rd(rd)
    , Vn(vn)
    , is_64(is64)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Rd = reg;
            return;
        }
        throw std::out_of_range("FMovFPToWXInst W/X register index out of range");
    }

    std::string to_asm() const override
    {
        return std::format("fmov {}{}, {}{}", is_64 ? "X" : "W", Rd, is_64 ? "D" : "S", Vn);
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_64 ? 0x9e660000u : 0x1e260000u;
        return static_cast<int>(base | (Vn << 5) | Rd);
    }
};

class FMADDScalarInst : public Instruction
{
    int Vd;
    int Vn;
    int Vm;
    int Va;
    bool is_64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::FMADD_SCALAR;
    OPCODE opcode() const override { return StaticOpcode; }
    FMADDScalarInst(int vd, int vn, int vm, int va, bool is64)
    : Vd(vd)
    , Vn(vn)
    , Vm(vm)
    , Va(va)
    , is_64(is64)
    {
    }

    std::string to_asm() const override
    {
        return std::format("fmadd {}{}, {}{}, {}{}, {}{}",
                           is_64 ? "D" : "S",
                           Vd,
                           is_64 ? "D" : "S",
                           Vn,
                           is_64 ? "D" : "S",
                           Vm,
                           is_64 ? "D" : "S",
                           Va);
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_64 ? 0x1f400000u : 0x1f000000u;
        return static_cast<int>(base | (Vm << 16) | (Va << 10) | (Vn << 5) | Vd);
    }
};

class LD1DInst : public Instruction
{
    int Zt;
    int Pg;
    int Xn;
    int Xm;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::LD1D;
    OPCODE opcode() const override { return StaticOpcode; }
    LD1DInst(int zt, int pg, int xn, int xm)
    : Zt(zt)
    , Pg(pg)
    , Xn(xn)
    , Xm(xm)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pg = reg;
            return;
        }
        throw std::out_of_range("LD1DInst P register index out of range");
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xn = reg;
            return;
        case 1:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("LD1DInst W/X register index out of range");
        }
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Zt = reg;
            return;
        }
        throw std::out_of_range("LD1DInst Z register index out of range");
    }

    std::string to_asm() const override
    {
        return std::format("ld1d {{ z{}.d }}, p{}/z, [x{}, x{}, lsl #3]", Zt, Pg, Xn, Xm);
    }

    int to_binary() const override
    {
        return static_cast<int>((0b10100101111u << 21) | (Zt << 0) | (Pg << 10) | (Xn << 5) | (Xm << 16) |
                                (0b010u << 13));
    }
};

class LD1WInst : public Instruction
{
    int Zt;
    int Pg;
    int Xn;
    int Xm;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::LD1W;
    OPCODE opcode() const override { return StaticOpcode; }
    LD1WInst(int zt, int pg, int xn, int xm)
    : Zt(zt)
    , Pg(pg)
    , Xn(xn)
    , Xm(xm)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pg = reg;
            return;
        }
        throw std::out_of_range("LD1WInst P register index out of range");
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xn = reg;
            return;
        case 1:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("LD1WInst W/X register index out of range");
        }
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Zt = reg;
            return;
        }
        throw std::out_of_range("LD1WInst Z register index out of range");
    }

    std::string to_asm() const override
    {
        return std::format("ld1w {{ z{}.s }}, p{}/z, [x{}, x{}, lsl #2]", Zt, Pg, Xn, Xm);
    }

    int to_binary() const override
    {
        return static_cast<int>((0b10100101010u << 21) | (Zt << 0) | (Pg << 10) | (Xn << 5) | (Xm << 16) |
                                (0b010u << 13));
    }
};

class BGEInst : public Instruction
{
    int imm19;
    std::string label;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::BGE;
    OPCODE opcode() const override { return StaticOpcode; }
    BGEInst(int imm, const std::string &lbl)
    : imm19(imm)
    , label(lbl)
    {
    }

    std::string to_asm() const override { return std::format("b.ge {}", label); }

    int to_binary() const override
    {
        const std::uint32_t imm = static_cast<std::uint32_t>(imm19) & 0x7ffffu;
        return static_cast<int>((0b01010100u << 24) | (imm << 5) | (0b01010u << 0));
    }

    const std::string &target() const { return label; }
    void set_imm19(int imm) { imm19 = imm; }
};

class BLTInst : public Instruction
{
    int imm19;
    std::string label;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::BLT;
    OPCODE opcode() const override { return StaticOpcode; }
    BLTInst(int imm, const std::string &lbl)
    : imm19(imm)
    , label(lbl)
    {
    }

    std::string to_asm() const override { return std::format("b.lt {}", label); }

    int to_binary() const override
    {
        const std::uint32_t imm = static_cast<std::uint32_t>(imm19) & 0x7ffffu;
        return static_cast<int>((0b01010100u << 24) | (imm << 5) | (0b01011u << 0));
    }

    const std::string &target() const { return label; }
    void set_imm19(int imm) { imm19 = imm; }
};

class MOVT2VInst : public Instruction
{
    int Zn;
    int Pg;
    int Ws;
    int Za;
    bool is_64;
    bool is_vertical;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::MOV_T2V;
    OPCODE opcode() const override { return StaticOpcode; }
    MOVT2VInst(int zn, int pg, int ws, int za, bool is64, bool isVertical)
    : Zn(zn)
    , Pg(pg)
    , Ws(ws)
    , Za(za)
    , is_64(is64)
    , is_vertical(isVertical)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pg = reg;
            return;
        }
        throw std::out_of_range("MOVT2VInst P register index out of range");
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Ws = reg;
            return;
        }
        throw std::out_of_range("MOVT2VInst W/X register index out of range");
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Zn = reg;
            return;
        }
        throw std::out_of_range("MOVT2VInst Z register index out of range");
    }

    std::string to_asm() const override
    {
        return std::format("mov za{}{}.{}[w{}, 0], p{}/m, z{}.{}",
                           Za,
                           is_vertical ? "v" : "h",
                           is_64 ? "d" : "s",
                           Ws,
                           Pg,
                           Zn,
                           is_64 ? "d" : "s");
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_64 ? 0xc0c00000u : 0xc0800000u;
        return static_cast<int>(base | (Zn << 5) | (Pg << 10) | ((Ws - 12) << 13) | (is_vertical ? (1u << 15) : 0u) |
                                (Za << (is_64 ? 1 : 2)));
    }
};

class MOVV2TInst : public Instruction
{
    int Zd;
    int Pg;
    int Ws;
    int Za;
    bool is_64;
    bool is_vertical;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::MOV_V2T;
    OPCODE opcode() const override { return StaticOpcode; }
    MOVV2TInst(int zd, int pg, int ws, int za, bool is64, bool isVertical)
    : Zd(zd)
    , Pg(pg)
    , Ws(ws)
    , Za(za)
    , is_64(is64)
    , is_vertical(isVertical)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pg = reg;
            return;
        }
        throw std::out_of_range("MOVV2TInst P register index out of range");
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Ws = reg;
            return;
        }
        throw std::out_of_range("MOVV2TInst W/X register index out of range");
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Zd = reg;
            return;
        }
        throw std::out_of_range("MOVV2TInst Z register index out of range");
    }

    std::string to_asm() const override
    {
        return std::format("mov z{}.{} , p{}/m, za{}{}.{}[w{}, 0]",
                           Zd,
                           is_64 ? "d" : "s",
                           Pg,
                           Za,
                           is_vertical ? "v" : "h",
                           is_64 ? "d" : "s",
                           Ws);
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_64 ? 0xc0c20000u : 0xc0820000u;
        return static_cast<int>(base | Zd | (Pg << 10) | ((Ws - 12) << 13) | (is_vertical ? (1u << 15) : 0u) |
                                (Za << (is_64 ? 6 : 7)));
    }
};

class ST1DInst : public Instruction
{
    int Zt;
    int Pg;
    int Xn;
    int Xm;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::ST1D;
    OPCODE opcode() const override { return StaticOpcode; }
    ST1DInst(int zt, int pg, int xn, int xm)
    : Zt(zt)
    , Pg(pg)
    , Xn(xn)
    , Xm(xm)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pg = reg;
            return;
        }
        throw std::out_of_range("ST1DInst P register index out of range");
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xn = reg;
            return;
        case 1:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("ST1DInst W/X register index out of range");
        }
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Zt = reg;
            return;
        }
        throw std::out_of_range("ST1DInst Z register index out of range");
    }

    std::string to_asm() const override
    {
        return std::format("st1d {{ z{}.d }}, p{}, [x{}, x{}, lsl #3]", Zt, Pg, Xn, Xm);
    }

    int to_binary() const override
    {
        return static_cast<int>((0b11100101111u << 21) | (Zt << 0) | (Pg << 10) | (Xn << 5) | (Xm << 16) |
                                (0b010u << 13));
    }
};

class ST1WInst : public Instruction
{
    int Zt;
    int Pg;
    int Xn;
    int Xm;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::ST1W;
    OPCODE opcode() const override { return StaticOpcode; }
    ST1WInst(int zt, int pg, int xn, int xm)
    : Zt(zt)
    , Pg(pg)
    , Xn(xn)
    , Xm(xm)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pg = reg;
            return;
        }
        throw std::out_of_range("ST1WInst P register index out of range");
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xn = reg;
            return;
        case 1:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("ST1WInst W/X register index out of range");
        }
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Zt = reg;
            return;
        }
        throw std::out_of_range("ST1WInst Z register index out of range");
    }

    std::string to_asm() const override
    {
        return std::format("st1w {{ z{}.s }}, p{}, [x{}, x{}, lsl #2]", Zt, Pg, Xn, Xm);
    }

    int to_binary() const override
    {
        return static_cast<int>((0b11100101010u << 21) | (Zt << 0) | (Pg << 10) | (Xn << 5) | (Xm << 16) |
                                (0b010u << 13));
    }
};

class EORInst : public Instruction
{
    int Pd;
    int Pg;
    int Pn;
    int Pm;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::EOR;
    OPCODE opcode() const override { return StaticOpcode; }
    EORInst(int pd, int pg, int pn, int pm)
    : Pd(pd)
    , Pg(pg)
    , Pn(pn)
    , Pm(pm)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Pd = reg;
            return;
        case 1:
            Pg = reg;
            return;
        case 2:
            Pn = reg;
            return;
        case 3:
            Pm = reg;
            return;
        default:
            throw std::out_of_range("EORInst P register index out of range");
        }
    }

    std::string to_asm() const override { return std::format("eor P{}.B, P{}/Z, P{}.B, P{}.B", Pd, Pg, Pn, Pm); }

    int to_binary() const override
    {
        return static_cast<int>((0b001001010000u << 20) | (Pd << 0) | (Pg << 10) | (Pn << 5) | (Pm << 16) |
                                (0b01u << 14) | (0b1u << 9) | (0b0u << 4));
    }
};

class PTUREInst : public Instruction
{
    int Pd;
    bool is_b64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::PTRUE;
    OPCODE opcode() const override { return StaticOpcode; }
    PTUREInst(int pd, bool is64)
    : Pd(pd)
    , is_b64(is64)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pd = reg;
            return;
        }
        throw std::out_of_range("PTUREInst P register index out of range");
    }

    std::string to_asm() const override { return std::format("ptrue p{}.{}", Pd, is_b64 ? "d" : "s"); }

    int to_binary() const override
    {
        return static_cast<int>((0b00100101u << 24) | (Pd << 0) | (0b011000111000111110u << 4) |
                                (is_b64 ? (0b11u << 22) : (0b10u << 22)));
    }
};

class PFALSEInst : public Instruction
{
    int Pd;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::PFALSE;
    OPCODE opcode() const override { return StaticOpcode; }
    PFALSEInst(int pd)
    : Pd(pd)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pd = reg;
            return;
        }
        throw std::out_of_range("PFALSEInst P register index out of range");
    }

    std::string to_asm() const override { return std::format("pfalse p{}.b", Pd); }

    int to_binary() const override { return static_cast<int>((0b0010010100011000111001000000u << 4) | (Pd << 0)); }
};

class WHILELTInst : public Instruction
{
    int Pd;
    int Xn;
    int Xm;
    bool is_b64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::WHILELT;
    OPCODE opcode() const override { return StaticOpcode; }
    WHILELTInst(int pd, int xn, int xm, bool is64)
    : Pd(pd)
    , Xn(xn)
    , Xm(xm)
    , is_b64(is64)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pd = reg;
            return;
        }
        throw std::out_of_range("WHILELTInst P register index out of range");
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Xn = reg;
            return;
        case 1:
            Xm = reg;
            return;
        default:
            throw std::out_of_range("WHILELTInst W/X register index out of range");
        }
    }

    std::string to_asm() const override
    {
        return std::format("whilelt p{}.{} , x{}, x{}", Pd, is_b64 ? "d" : "s", Xn, Xm);
    }

    int to_binary() const override
    {
        return static_cast<int>((0b00100101u << 24) | (is_b64 ? (0b11u << 22) : (0b10u << 22)) | (0b1u << 21) |
                                (Xm << 16) | (0b000101u << 10) | (Xn << 5) | (Pd << 0));
    }
};

class SELInst : public Instruction
{
    int Zd;
    int Pv;
    int Zn;
    int Zm;
    bool is_b64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::SEL;
    OPCODE opcode() const override { return StaticOpcode; }
    SELInst(int zd, int pv, int zn, int zm, bool is64)
    : Zd(zd)
    , Pv(pv)
    , Zn(zn)
    , Zm(zm)
    , is_b64(is64)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pv = reg;
            return;
        }
        throw std::out_of_range("SELInst P register index out of range");
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Zd = reg;
            return;
        case 1:
            Zn = reg;
            return;
        case 2:
            Zm = reg;
            return;
        default:
            throw std::out_of_range("SELInst Z register index out of range");
        }
    }

    std::string to_asm() const override
    {
        return std::format("sel z{}.{} , p{}, z{}.{} , z{}.{}",
                           Zd,
                           is_b64 ? "d" : "s",
                           Pv,
                           Zn,
                           is_b64 ? "d" : "s",
                           Zm,
                           is_b64 ? "d" : "s");
    }

    int to_binary() const override
    {
        return static_cast<int>((0b00000101u << 24) | (is_b64 ? (0b11u << 22) : (0b10u << 22)) | (0b1u << 21) |
                                (Zm << 16) | (0b11u << 14) | (Pv << 10) | (Zn << 5) | (Zd << 0));
    }
};

class FMLASVEInst : public Instruction
{
    int Zda;
    int Pg;
    int Zn;
    int Zm;
    bool is_b64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::FMLA_SVE;
    OPCODE opcode() const override { return StaticOpcode; }
    FMLASVEInst(int zda, int pg, int zn, int zm, bool is64)
    : Zda(zda)
    , Pg(pg)
    , Zn(zn)
    , Zm(zm)
    , is_b64(is64)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Pg = reg;
            return;
        }
        throw std::out_of_range("FMLASVEInst P register index out of range");
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Zda = reg;
            return;
        case 1:
            Zn = reg;
            return;
        case 2:
            Zm = reg;
            return;
        default:
            throw std::out_of_range("FMLASVEInst Z register index out of range");
        }
    }

    std::string to_asm() const override
    {
        return std::format("fmla z{}.{} , p{}/m, z{}.{} , z{}.{}",
                           Zda,
                           is_b64 ? "d" : "s",
                           Pg,
                           Zn,
                           is_b64 ? "d" : "s",
                           Zm,
                           is_b64 ? "d" : "s");
    }

    int to_binary() const override
    {
        return static_cast<int>((0b01100101u << 24) | (is_b64 ? (0b11u << 22) : (0b10u << 22)) | (0b1u << 21) |
                                (Zm << 16) | (0b000u << 13) | (Pg << 10) | (Zn << 5) | (Zda << 0));
    }
};

class FMLASMEInst : public Instruction
{
    int Wv;
    int Zn1;
    int Zn2;
    int Zm1;
    int Zm2;
    bool is_b64;
    bool is_VG2;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::FMLA_SME;
    OPCODE opcode() const override { return StaticOpcode; }
    FMLASMEInst(int wv, int zn1, int zn2, int zm1, int zm2, bool is64, bool isVG2)
    : Wv(wv)
    , Zn1(zn1)
    , Zn2(zn2)
    , Zm1(zm1)
    , Zm2(zm2)
    , is_b64(is64)
    , is_VG2(isVG2)
    {
    }

    void set_wx_reg(std::size_t index, int reg) override
    {
        if (index == 0)
        {
            Wv = reg;
            return;
        }
        throw std::out_of_range("FMLASMEInst W/X register index out of range");
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Zn1 = reg;
            return;
        case 1:
            Zn2 = reg;
            return;
        case 2:
            Zm1 = reg;
            return;
        case 3:
            Zm2 = reg;
            return;
        default:
            throw std::out_of_range("FMLASMEInst Z register index out of range");
        }
    }

    std::string to_asm() const override
    {
        if (is_VG2)
        {
            return std::format("fmla za.{}[w{}, 0, vgx2], {{ z{}.{}, z{}.{} }}, {{ z{}.{}, z{}.{} }}",
                               is_b64 ? "d" : "s",
                               Wv,
                               Zn1,
                               is_b64 ? "d" : "s",
                               Zn2,
                               is_b64 ? "d" : "s",
                               Zm1,
                               is_b64 ? "d" : "s",
                               Zm2,
                               is_b64 ? "d" : "s");
        }

        return std::format("fmla za.{}[w{}, 0, vgx4], {{ z{}.{} - z{}.{} }}, {{ z{}.{} - z{}.{} }}",
                           is_b64 ? "d" : "s",
                           Wv,
                           Zn1,
                           is_b64 ? "d" : "s",
                           Zn2,
                           is_b64 ? "d" : "s",
                           Zm1,
                           is_b64 ? "d" : "s",
                           Zm2,
                           is_b64 ? "d" : "s");
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_VG2 ? (is_b64 ? 0xc1e01800u : 0xc1a01800u) : (is_b64 ? 0xc1e11800u : 0xc1a11800u);
        return static_cast<int>(base | (Zm1 << 16) | ((Wv - 8) << 13) | (Zn1 << 5));
    }
};

class FMOPAInst : public Instruction
{
    int Zada;
    int Pn;
    int Pm;
    int Zn;
    int Zm;
    bool is_64;

public:
    static constexpr OPCODE StaticOpcode = OPCODE::FMOPA;
    OPCODE opcode() const override { return StaticOpcode; }
    FMOPAInst(int zada, int pn, int pm, int zn, int zm, bool is64)
    : Zada(zada)
    , Pn(pn)
    , Pm(pm)
    , Zn(zn)
    , Zm(zm)
    , is_64(is64)
    {
    }

    void set_p_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Pn = reg;
            return;
        case 1:
            Pm = reg;
            return;
        default:
            throw std::out_of_range("FMOPAInst P register index out of range");
        }
    }

    void set_z_reg(std::size_t index, int reg) override
    {
        switch (index)
        {
        case 0:
            Zada = reg;
            return;
        case 1:
            Zn = reg;
            return;
        case 2:
            Zm = reg;
            return;
        default:
            throw std::out_of_range("FMOPAInst Z register index out of range");
        }
    }

    std::string to_asm() const override
    {
        return std::format("fmopa za{}.{} , p{}/m, p{}/m, z{}.{} , z{}.{}",
                           Zada,
                           is_64 ? "d" : "s",
                           Pn,
                           Pm,
                           Zn,
                           is_64 ? "d" : "s",
                           Zm,
                           is_64 ? "d" : "s");
    }

    int to_binary() const override
    {
        const std::uint32_t base = is_64 ? 0x80c00000u : 0x80800000u;
        return static_cast<int>(base | (Zm << 16) | (Pm << 13) | (Pn << 10) | (Zn << 5) | Zada);
    }
};

class RETInst : public Instruction
{
public:
    static constexpr OPCODE StaticOpcode = OPCODE::RET;
    OPCODE opcode() const override { return StaticOpcode; }
    std::string to_asm() const override { return "ret"; }

    int to_binary() const override { return static_cast<int>(0xd65f03c0u); }
};

} // namespace ARM
