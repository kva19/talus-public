// include/strata/core/dflash_prefill.hpp - Talus: prefill-side DFlash2 tap capture (v2).
//
// The prefill trunk walks layers with the chunk's mixed representation final at each boundary;
// these helpers stash the LAST rows of the final chunk into the drafter's persistent tap slots
// (allocated in dflash_drafter.cpp). The generation-start inject in generate.cpp then projects
// them into the drafter's KV ring, so the drafter sees the prompt tail before its first draft.
#pragma once

#include <cstdint>

namespace strata::core {

/// True when the armed drafter reads target layer `layer_1based` (and is armed at all).
bool dflash_prefill_wants(int64_t layer_1based);

/// Copy the LAST min(T, slot_rows) rows of `mixed` (T rows of n_embd, device) into the slot.
void dflash_prefill_tap(int64_t layer_1based, const float* mixed, int T, int n_embd, void* stream);

}  // namespace strata::core
