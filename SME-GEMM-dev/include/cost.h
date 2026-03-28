#pragma once

#if __has_include("cost_config.h")
#include "cost_config.h"
#endif

#ifndef COSTCFG_SVE_FMLA_COST
#define COSTCFG_SVE_FMLA_COST 1.00
#endif
#ifndef COSTCFG_SME_MOPA_COST
#define COSTCFG_SME_MOPA_COST 1.05
#endif
#ifndef COSTCFG_SME2_FMLA_COST
#define COSTCFG_SME2_FMLA_COST 1.20
#endif
#ifndef COSTCFG_SVE_LOAD_COST
#define COSTCFG_SVE_LOAD_COST 0.30
#endif
#ifndef COSTCFG_SVE_CONCAT_COST
#define COSTCFG_SVE_CONCAT_COST 0.20
#endif
#ifndef COSTCFG_SVE_STORE_COST
#define COSTCFG_SVE_STORE_COST 0.35
#endif
#ifndef COSTCFG_SME_MOVA_COST
#define COSTCFG_SME_MOVA_COST 0.45
#endif
#ifndef COSTCFG_SCALAR_LOAD_COST
#define COSTCFG_SCALAR_LOAD_COST 0.40
#endif
#ifndef COSTCFG_SCALAR_STORE_COST
#define COSTCFG_SCALAR_STORE_COST 0.45
#endif
#ifndef COSTCFG_SCALAR_COMPUTE_COST
#define COSTCFG_SCALAR_COMPUTE_COST 1.60
#endif
#ifndef COSTCFG_SVE_INSNS_COVERED_PER_MOPA
#define COSTCFG_SVE_INSNS_COVERED_PER_MOPA 3
#endif
#ifndef COSTCFG_SME_MOVE_INSNS_COVERED_PER_MOPA
#define COSTCFG_SME_MOVE_INSNS_COVERED_PER_MOPA 1
#endif
#ifndef COSTCFG_SVE_LOAD_INSNS_COVERED_PER_MOPA
#define COSTCFG_SVE_LOAD_INSNS_COVERED_PER_MOPA 1
#endif
#ifndef COSTCFG_SVE_STORE_INSNS_COVERED_PER_MOPA
#define COSTCFG_SVE_STORE_INSNS_COVERED_PER_MOPA 1
#endif
#ifndef COSTCFG_SVE_CONCAT_INSNS_COVERED_PER_MOPA
#define COSTCFG_SVE_CONCAT_INSNS_COVERED_PER_MOPA 1
#endif
#ifndef COSTCFG_MOPA_CHAIN_WIDTH
#define COSTCFG_MOPA_CHAIN_WIDTH 8
#endif
#ifndef COSTCFG_MOPA_CHAIN_HEAD
#define COSTCFG_MOPA_CHAIN_HEAD 2
#endif
#ifndef COSTCFG_MEM_COMPUTE_OVERLAP
#define COSTCFG_MEM_COMPUTE_OVERLAP 0.25
#endif
#ifndef COSTCFG_ENABLE_SME2
#define COSTCFG_ENABLE_SME2 0
#endif

// Per-instruction relative latency/throughput cost. These numbers are tuned by
// microbenchmarks and can be adjusted per machine.
constexpr double SVE_FMLA_COST = COSTCFG_SVE_FMLA_COST;
constexpr double SME_MOPA_COST = COSTCFG_SME_MOPA_COST;
constexpr double SME2_FMLA_COST = COSTCFG_SME2_FMLA_COST;
constexpr double SVE_LOAD_COST = COSTCFG_SVE_LOAD_COST;
constexpr double SVE_CONCAT_COST = COSTCFG_SVE_CONCAT_COST;
constexpr double SVE_STORE_COST = COSTCFG_SVE_STORE_COST;
constexpr double SME_MOVA_COST = COSTCFG_SME_MOVA_COST;
constexpr double SCALAR_LOAD_COST = COSTCFG_SCALAR_LOAD_COST;
constexpr double SCALAR_STORE_COST = COSTCFG_SCALAR_STORE_COST;
constexpr double SCALAR_COMPUTE_COST = COSTCFG_SCALAR_COMPUTE_COST;

// Overlap is modeled by coverage, not by percentage blending.
// One MOPA can hide a few SVE compute instructions on this machine.
constexpr int SVE_INSNS_COVERED_PER_MOPA = COSTCFG_SVE_INSNS_COVERED_PER_MOPA;
// ZA movement that can be hidden behind one MOPA.
constexpr int SME_MOVE_INSNS_COVERED_PER_MOPA = COSTCFG_SME_MOVE_INSNS_COVERED_PER_MOPA;
// SVE memory ops that can be hidden by one MOPA.
constexpr int SVE_LOAD_INSNS_COVERED_PER_MOPA = COSTCFG_SVE_LOAD_INSNS_COVERED_PER_MOPA;
constexpr int SVE_STORE_INSNS_COVERED_PER_MOPA = COSTCFG_SVE_STORE_INSNS_COVERED_PER_MOPA;
// Predicate/concat style SVE glue ops hidden by one MOPA.
constexpr int SVE_CONCAT_INSNS_COVERED_PER_MOPA = COSTCFG_SVE_CONCAT_INSNS_COVERED_PER_MOPA;
// MOPA-to-MOPA pipeline overlap: after warmup, every group of this width can
// hide roughly one MOPA issue slot.
constexpr int MOPA_CHAIN_WIDTH = COSTCFG_MOPA_CHAIN_WIDTH;
// Initial MOPA slots paid before steady overlap takes effect.
constexpr int MOPA_CHAIN_HEAD = COSTCFG_MOPA_CHAIN_HEAD;

// Memory and compute can overlap partially depending on kernel shape.
constexpr double MEM_COMPUTE_OVERLAP = COSTCFG_MEM_COMPUTE_OVERLAP;

// SME2 is controlled by generated config and additionally guarded by runtime CPU checks.
constexpr bool COSTMODEL_ENABLE_SME2 = (COSTCFG_ENABLE_SME2 != 0);
