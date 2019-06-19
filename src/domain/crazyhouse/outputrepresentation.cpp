/*
 * CrazyAra, a deep learning chess variant engine
 * Copyright (C) 2018 Johannes Czech, Moritz Willig, Alena Beyer
 * Copyright (C) 2019 Johannes Czech
 *
 * CrazyAra is free software: You can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * @file: outputrepresentation.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include "outputrepresentation.h"
#include "types.h"

// TODO: Change this later to blaze::HybridVector<float, MAX_NB_LEGAL_MOVES>
void get_probs_of_move_list(const NDArray policyProb, const std::vector<Move> &legalMoves, Color sideToMove, bool normalize, DynamicVector<float> &policyProbSmall)
{
    // allocate sufficient memory
//    DynamicVector<float> policyProbSmall(legalMoves.size());
    policyProbSmall.resize(legalMoves.size());
    policyProb.WaitToRead();

    size_t idx;
    if (sideToMove == WHITE) {
        for (size_t mvIdx = 0; mvIdx < legalMoves.size(); ++mvIdx)
        {
            // find the according index in the vector
            idx = MV_LOOKUP[legalMoves[mvIdx]];
            // set the right prob value
            policyProbSmall[mvIdx] = policyProb.At(0, idx);
        }
    }
    else {
        for (size_t mvIdx = 0; mvIdx < legalMoves.size(); ++mvIdx)
        {
            // use the mirrored look-up table instead
            idx = MV_LOOKUP_MIRRORED[legalMoves[mvIdx]];

            // set the right prob value
            policyProbSmall[mvIdx] = policyProb.At(0, idx);
        }
    }

    if (normalize) {
        policyProbSmall /= sum(policyProbSmall);
    }
//    return policyProbSmall;
}


// https://helloacm.com/how-to-implement-the-sgn-function-in-c/
template <class T>
inline int
sgn(T v) {
    return (v > T(0)) - (v < T(0));
}

int value_to_centipawn(float value)
{
    if (std::abs(value) >= 1) {
        // return a constant if the given value is 1 (otherwise log will result in infinity)
        return sgn(value) * 9999;
    }
    // use logarithmic scaling with basis 1.1 as a pseudo centipawn conversion
    return int(-(sgn(value) * std::log(1.0f - std::abs(value)) / std::log(1.2f)) * 100.0f);
}

