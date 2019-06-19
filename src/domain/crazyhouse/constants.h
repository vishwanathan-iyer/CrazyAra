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
 * @file: constants.h
 * Created on 13.05.2019
 * @author: queensgambit
 *
 * Definition of all constants for the game of Crazyhouse
 *
 * ! DO NOT CHANGE THE LABEL LIST OTHERWISE YOU WILL BREAK THE MOVE MAPPING OF THE NETWORK !
 *
 * This file contains the description of the policy vector which is predicted by the neural network and
 * constants for designing the input planes representation.
 * Each index of the array LABEL array corresponds to a unique move in uci-representation
 */

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>
#include <unordered_map>
#include "types.h"
#include "../../sfutil.h"
#include <iostream>

// Define the board size
const int BOARD_WIDTH = 8;
const int BOARD_HEIGHT = 8;
const int NB_SQUARES = BOARD_WIDTH * BOARD_HEIGHT;

// define two constants indicating the number of channels for the input plane presentation
const int NB_CHANNELS_POS = 27;
const int NB_CHANNELS_CONST = 7;
const int NB_CHANNELS_TOTAL = NB_CHANNELS_POS + NB_CHANNELS_CONST;
const int NB_VALUES_TOTAL = NB_CHANNELS_TOTAL*BOARD_HEIGHT*BOARD_WIDTH;

// define the number of different pieces one can have in his pocket (the king is excluded)
const int POCKETS_SIZE_PIECE_TYPE = 5;

//  (this used for normalization the input planes and setting an appropriate integer representation (e.g. int16)
// these are defined as float to avoid integer division
const float MAX_NB_PRISONERS = 32;  // define the maximum number of pieces of each type in a pocket
const float MAX_FULL_MOVE_COUNTER = 500;  // 500 was set as the max number of total moves
const float MAX_NB_NO_PROGRESS = 40;  // after 40 moves of no progress the 40 moves rule for draw applies

// this is used during the MCTS for static memory allocation (for Crazyhouse this number will never be reached)
// definition for the maximum number of legal moves in an arbitrary board position
const unsigned long MAX_NB_LEGAL_MOVES = 512;

// Policy Vector Description:
// (Note that this vector does only work for the white player.
// For the black player you have ot mirror the move afterwards by using mirror_move())

const std::string LABELS[] = {
    "a1b1",
    "a1c1",
    "a1d1",
    "a1e1",
    "a1f1",
    "a1g1",
    "a1h1",
    "a1a2",
    "a1a3",
    "a1a4",
    "a1a5",
    "a1a6",
    "a1a7",
    "a1a8",
    "a1b2",
    "a1c3",
    "a1d4",
    "a1e5",
    "a1f6",
    "a1g7",
    "a1h8",
    "a1c2",
    "a1b3",
    "a2b2",
    "a2c2",
    "a2d2",
    "a2e2",
    "a2f2",
    "a2g2",
    "a2h2",
    "a2a1",
    "a2a3",
    "a2a4",
    "a2a5",
    "a2a6",
    "a2a7",
    "a2a8",
    "a2b3",
    "a2c4",
    "a2d5",
    "a2e6",
    "a2f7",
    "a2g8",
    "a2b1",
    "a2c1",
    "a2c3",
    "a2b4",
    "a3b3",
    "a3c3",
    "a3d3",
    "a3e3",
    "a3f3",
    "a3g3",
    "a3h3",
    "a3a1",
    "a3a2",
    "a3a4",
    "a3a5",
    "a3a6",
    "a3a7",
    "a3a8",
    "a3b4",
    "a3c5",
    "a3d6",
    "a3e7",
    "a3f8",
    "a3b2",
    "a3c1",
    "a3b1",
    "a3c2",
    "a3c4",
    "a3b5",
    "a4b4",
    "a4c4",
    "a4d4",
    "a4e4",
    "a4f4",
    "a4g4",
    "a4h4",
    "a4a1",
    "a4a2",
    "a4a3",
    "a4a5",
    "a4a6",
    "a4a7",
    "a4a8",
    "a4b5",
    "a4c6",
    "a4d7",
    "a4e8",
    "a4b3",
    "a4c2",
    "a4d1",
    "a4b2",
    "a4c3",
    "a4c5",
    "a4b6",
    "a5b5",
    "a5c5",
    "a5d5",
    "a5e5",
    "a5f5",
    "a5g5",
    "a5h5",
    "a5a1",
    "a5a2",
    "a5a3",
    "a5a4",
    "a5a6",
    "a5a7",
    "a5a8",
    "a5b6",
    "a5c7",
    "a5d8",
    "a5b4",
    "a5c3",
    "a5d2",
    "a5e1",
    "a5b3",
    "a5c4",
    "a5c6",
    "a5b7",
    "a6b6",
    "a6c6",
    "a6d6",
    "a6e6",
    "a6f6",
    "a6g6",
    "a6h6",
    "a6a1",
    "a6a2",
    "a6a3",
    "a6a4",
    "a6a5",
    "a6a7",
    "a6a8",
    "a6b7",
    "a6c8",
    "a6b5",
    "a6c4",
    "a6d3",
    "a6e2",
    "a6f1",
    "a6b4",
    "a6c5",
    "a6c7",
    "a6b8",
    "a7b7",
    "a7c7",
    "a7d7",
    "a7e7",
    "a7f7",
    "a7g7",
    "a7h7",
    "a7a1",
    "a7a2",
    "a7a3",
    "a7a4",
    "a7a5",
    "a7a6",
    "a7a8",
    "a7b8",
    "a7b6",
    "a7c5",
    "a7d4",
    "a7e3",
    "a7f2",
    "a7g1",
    "a7b5",
    "a7c6",
    "a7c8",
    "a8b8",
    "a8c8",
    "a8d8",
    "a8e8",
    "a8f8",
    "a8g8",
    "a8h8",
    "a8a1",
    "a8a2",
    "a8a3",
    "a8a4",
    "a8a5",
    "a8a6",
    "a8a7",
    "a8b7",
    "a8c6",
    "a8d5",
    "a8e4",
    "a8f3",
    "a8g2",
    "a8h1",
    "a8b6",
    "a8c7",
    "b1a1",
    "b1c1",
    "b1d1",
    "b1e1",
    "b1f1",
    "b1g1",
    "b1h1",
    "b1b2",
    "b1b3",
    "b1b4",
    "b1b5",
    "b1b6",
    "b1b7",
    "b1b8",
    "b1c2",
    "b1d3",
    "b1e4",
    "b1f5",
    "b1g6",
    "b1h7",
    "b1a2",
    "b1a3",
    "b1d2",
    "b1c3",
    "b2a2",
    "b2c2",
    "b2d2",
    "b2e2",
    "b2f2",
    "b2g2",
    "b2h2",
    "b2b1",
    "b2b3",
    "b2b4",
    "b2b5",
    "b2b6",
    "b2b7",
    "b2b8",
    "b2a1",
    "b2c3",
    "b2d4",
    "b2e5",
    "b2f6",
    "b2g7",
    "b2h8",
    "b2a3",
    "b2c1",
    "b2d1",
    "b2a4",
    "b2d3",
    "b2c4",
    "b3a3",
    "b3c3",
    "b3d3",
    "b3e3",
    "b3f3",
    "b3g3",
    "b3h3",
    "b3b1",
    "b3b2",
    "b3b4",
    "b3b5",
    "b3b6",
    "b3b7",
    "b3b8",
    "b3a2",
    "b3c4",
    "b3d5",
    "b3e6",
    "b3f7",
    "b3g8",
    "b3a4",
    "b3c2",
    "b3d1",
    "b3a1",
    "b3c1",
    "b3d2",
    "b3a5",
    "b3d4",
    "b3c5",
    "b4a4",
    "b4c4",
    "b4d4",
    "b4e4",
    "b4f4",
    "b4g4",
    "b4h4",
    "b4b1",
    "b4b2",
    "b4b3",
    "b4b5",
    "b4b6",
    "b4b7",
    "b4b8",
    "b4a3",
    "b4c5",
    "b4d6",
    "b4e7",
    "b4f8",
    "b4a5",
    "b4c3",
    "b4d2",
    "b4e1",
    "b4a2",
    "b4c2",
    "b4d3",
    "b4a6",
    "b4d5",
    "b4c6",
    "b5a5",
    "b5c5",
    "b5d5",
    "b5e5",
    "b5f5",
    "b5g5",
    "b5h5",
    "b5b1",
    "b5b2",
    "b5b3",
    "b5b4",
    "b5b6",
    "b5b7",
    "b5b8",
    "b5a4",
    "b5c6",
    "b5d7",
    "b5e8",
    "b5a6",
    "b5c4",
    "b5d3",
    "b5e2",
    "b5f1",
    "b5a3",
    "b5c3",
    "b5d4",
    "b5a7",
    "b5d6",
    "b5c7",
    "b6a6",
    "b6c6",
    "b6d6",
    "b6e6",
    "b6f6",
    "b6g6",
    "b6h6",
    "b6b1",
    "b6b2",
    "b6b3",
    "b6b4",
    "b6b5",
    "b6b7",
    "b6b8",
    "b6a5",
    "b6c7",
    "b6d8",
    "b6a7",
    "b6c5",
    "b6d4",
    "b6e3",
    "b6f2",
    "b6g1",
    "b6a4",
    "b6c4",
    "b6d5",
    "b6a8",
    "b6d7",
    "b6c8",
    "b7a7",
    "b7c7",
    "b7d7",
    "b7e7",
    "b7f7",
    "b7g7",
    "b7h7",
    "b7b1",
    "b7b2",
    "b7b3",
    "b7b4",
    "b7b5",
    "b7b6",
    "b7b8",
    "b7a6",
    "b7c8",
    "b7a8",
    "b7c6",
    "b7d5",
    "b7e4",
    "b7f3",
    "b7g2",
    "b7h1",
    "b7a5",
    "b7c5",
    "b7d6",
    "b7d8",
    "b8a8",
    "b8c8",
    "b8d8",
    "b8e8",
    "b8f8",
    "b8g8",
    "b8h8",
    "b8b1",
    "b8b2",
    "b8b3",
    "b8b4",
    "b8b5",
    "b8b6",
    "b8b7",
    "b8a7",
    "b8c7",
    "b8d6",
    "b8e5",
    "b8f4",
    "b8g3",
    "b8h2",
    "b8a6",
    "b8c6",
    "b8d7",
    "c1a1",
    "c1b1",
    "c1d1",
    "c1e1",
    "c1f1",
    "c1g1",
    "c1h1",
    "c1c2",
    "c1c3",
    "c1c4",
    "c1c5",
    "c1c6",
    "c1c7",
    "c1c8",
    "c1d2",
    "c1e3",
    "c1f4",
    "c1g5",
    "c1h6",
    "c1a3",
    "c1b2",
    "c1a2",
    "c1b3",
    "c1e2",
    "c1d3",
    "c2a2",
    "c2b2",
    "c2d2",
    "c2e2",
    "c2f2",
    "c2g2",
    "c2h2",
    "c2c1",
    "c2c3",
    "c2c4",
    "c2c5",
    "c2c6",
    "c2c7",
    "c2c8",
    "c2b1",
    "c2d3",
    "c2e4",
    "c2f5",
    "c2g6",
    "c2h7",
    "c2a4",
    "c2b3",
    "c2d1",
    "c2a1",
    "c2a3",
    "c2e1",
    "c2b4",
    "c2e3",
    "c2d4",
    "c3a3",
    "c3b3",
    "c3d3",
    "c3e3",
    "c3f3",
    "c3g3",
    "c3h3",
    "c3c1",
    "c3c2",
    "c3c4",
    "c3c5",
    "c3c6",
    "c3c7",
    "c3c8",
    "c3a1",
    "c3b2",
    "c3d4",
    "c3e5",
    "c3f6",
    "c3g7",
    "c3h8",
    "c3a5",
    "c3b4",
    "c3d2",
    "c3e1",
    "c3a2",
    "c3b1",
    "c3a4",
    "c3d1",
    "c3e2",
    "c3b5",
    "c3e4",
    "c3d5",
    "c4a4",
    "c4b4",
    "c4d4",
    "c4e4",
    "c4f4",
    "c4g4",
    "c4h4",
    "c4c1",
    "c4c2",
    "c4c3",
    "c4c5",
    "c4c6",
    "c4c7",
    "c4c8",
    "c4a2",
    "c4b3",
    "c4d5",
    "c4e6",
    "c4f7",
    "c4g8",
    "c4a6",
    "c4b5",
    "c4d3",
    "c4e2",
    "c4f1",
    "c4a3",
    "c4b2",
    "c4a5",
    "c4d2",
    "c4e3",
    "c4b6",
    "c4e5",
    "c4d6",
    "c5a5",
    "c5b5",
    "c5d5",
    "c5e5",
    "c5f5",
    "c5g5",
    "c5h5",
    "c5c1",
    "c5c2",
    "c5c3",
    "c5c4",
    "c5c6",
    "c5c7",
    "c5c8",
    "c5a3",
    "c5b4",
    "c5d6",
    "c5e7",
    "c5f8",
    "c5a7",
    "c5b6",
    "c5d4",
    "c5e3",
    "c5f2",
    "c5g1",
    "c5a4",
    "c5b3",
    "c5a6",
    "c5d3",
    "c5e4",
    "c5b7",
    "c5e6",
    "c5d7",
    "c6a6",
    "c6b6",
    "c6d6",
    "c6e6",
    "c6f6",
    "c6g6",
    "c6h6",
    "c6c1",
    "c6c2",
    "c6c3",
    "c6c4",
    "c6c5",
    "c6c7",
    "c6c8",
    "c6a4",
    "c6b5",
    "c6d7",
    "c6e8",
    "c6a8",
    "c6b7",
    "c6d5",
    "c6e4",
    "c6f3",
    "c6g2",
    "c6h1",
    "c6a5",
    "c6b4",
    "c6a7",
    "c6d4",
    "c6e5",
    "c6b8",
    "c6e7",
    "c6d8",
    "c7a7",
    "c7b7",
    "c7d7",
    "c7e7",
    "c7f7",
    "c7g7",
    "c7h7",
    "c7c1",
    "c7c2",
    "c7c3",
    "c7c4",
    "c7c5",
    "c7c6",
    "c7c8",
    "c7a5",
    "c7b6",
    "c7d8",
    "c7b8",
    "c7d6",
    "c7e5",
    "c7f4",
    "c7g3",
    "c7h2",
    "c7a6",
    "c7b5",
    "c7a8",
    "c7d5",
    "c7e6",
    "c7e8",
    "c8a8",
    "c8b8",
    "c8d8",
    "c8e8",
    "c8f8",
    "c8g8",
    "c8h8",
    "c8c1",
    "c8c2",
    "c8c3",
    "c8c4",
    "c8c5",
    "c8c6",
    "c8c7",
    "c8a6",
    "c8b7",
    "c8d7",
    "c8e6",
    "c8f5",
    "c8g4",
    "c8h3",
    "c8a7",
    "c8b6",
    "c8d6",
    "c8e7",
    "d1a1",
    "d1b1",
    "d1c1",
    "d1e1",
    "d1f1",
    "d1g1",
    "d1h1",
    "d1d2",
    "d1d3",
    "d1d4",
    "d1d5",
    "d1d6",
    "d1d7",
    "d1d8",
    "d1e2",
    "d1f3",
    "d1g4",
    "d1h5",
    "d1a4",
    "d1b3",
    "d1c2",
    "d1b2",
    "d1c3",
    "d1f2",
    "d1e3",
    "d2a2",
    "d2b2",
    "d2c2",
    "d2e2",
    "d2f2",
    "d2g2",
    "d2h2",
    "d2d1",
    "d2d3",
    "d2d4",
    "d2d5",
    "d2d6",
    "d2d7",
    "d2d8",
    "d2c1",
    "d2e3",
    "d2f4",
    "d2g5",
    "d2h6",
    "d2a5",
    "d2b4",
    "d2c3",
    "d2e1",
    "d2b1",
    "d2b3",
    "d2f1",
    "d2c4",
    "d2f3",
    "d2e4",
    "d3a3",
    "d3b3",
    "d3c3",
    "d3e3",
    "d3f3",
    "d3g3",
    "d3h3",
    "d3d1",
    "d3d2",
    "d3d4",
    "d3d5",
    "d3d6",
    "d3d7",
    "d3d8",
    "d3b1",
    "d3c2",
    "d3e4",
    "d3f5",
    "d3g6",
    "d3h7",
    "d3a6",
    "d3b5",
    "d3c4",
    "d3e2",
    "d3f1",
    "d3b2",
    "d3c1",
    "d3b4",
    "d3e1",
    "d3f2",
    "d3c5",
    "d3f4",
    "d3e5",
    "d4a4",
    "d4b4",
    "d4c4",
    "d4e4",
    "d4f4",
    "d4g4",
    "d4h4",
    "d4d1",
    "d4d2",
    "d4d3",
    "d4d5",
    "d4d6",
    "d4d7",
    "d4d8",
    "d4a1",
    "d4b2",
    "d4c3",
    "d4e5",
    "d4f6",
    "d4g7",
    "d4h8",
    "d4a7",
    "d4b6",
    "d4c5",
    "d4e3",
    "d4f2",
    "d4g1",
    "d4b3",
    "d4c2",
    "d4b5",
    "d4e2",
    "d4f3",
    "d4c6",
    "d4f5",
    "d4e6",
    "d5a5",
    "d5b5",
    "d5c5",
    "d5e5",
    "d5f5",
    "d5g5",
    "d5h5",
    "d5d1",
    "d5d2",
    "d5d3",
    "d5d4",
    "d5d6",
    "d5d7",
    "d5d8",
    "d5a2",
    "d5b3",
    "d5c4",
    "d5e6",
    "d5f7",
    "d5g8",
    "d5a8",
    "d5b7",
    "d5c6",
    "d5e4",
    "d5f3",
    "d5g2",
    "d5h1",
    "d5b4",
    "d5c3",
    "d5b6",
    "d5e3",
    "d5f4",
    "d5c7",
    "d5f6",
    "d5e7",
    "d6a6",
    "d6b6",
    "d6c6",
    "d6e6",
    "d6f6",
    "d6g6",
    "d6h6",
    "d6d1",
    "d6d2",
    "d6d3",
    "d6d4",
    "d6d5",
    "d6d7",
    "d6d8",
    "d6a3",
    "d6b4",
    "d6c5",
    "d6e7",
    "d6f8",
    "d6b8",
    "d6c7",
    "d6e5",
    "d6f4",
    "d6g3",
    "d6h2",
    "d6b5",
    "d6c4",
    "d6b7",
    "d6e4",
    "d6f5",
    "d6c8",
    "d6f7",
    "d6e8",
    "d7a7",
    "d7b7",
    "d7c7",
    "d7e7",
    "d7f7",
    "d7g7",
    "d7h7",
    "d7d1",
    "d7d2",
    "d7d3",
    "d7d4",
    "d7d5",
    "d7d6",
    "d7d8",
    "d7a4",
    "d7b5",
    "d7c6",
    "d7e8",
    "d7c8",
    "d7e6",
    "d7f5",
    "d7g4",
    "d7h3",
    "d7b6",
    "d7c5",
    "d7b8",
    "d7e5",
    "d7f6",
    "d7f8",
    "d8a8",
    "d8b8",
    "d8c8",
    "d8e8",
    "d8f8",
    "d8g8",
    "d8h8",
    "d8d1",
    "d8d2",
    "d8d3",
    "d8d4",
    "d8d5",
    "d8d6",
    "d8d7",
    "d8a5",
    "d8b6",
    "d8c7",
    "d8e7",
    "d8f6",
    "d8g5",
    "d8h4",
    "d8b7",
    "d8c6",
    "d8e6",
    "d8f7",
    "e1a1",
    "e1b1",
    "e1c1",
    "e1d1",
    "e1f1",
    "e1g1",
    "e1h1",
    "e1e2",
    "e1e3",
    "e1e4",
    "e1e5",
    "e1e6",
    "e1e7",
    "e1e8",
    "e1f2",
    "e1g3",
    "e1h4",
    "e1a5",
    "e1b4",
    "e1c3",
    "e1d2",
    "e1c2",
    "e1d3",
    "e1g2",
    "e1f3",
    "e2a2",
    "e2b2",
    "e2c2",
    "e2d2",
    "e2f2",
    "e2g2",
    "e2h2",
    "e2e1",
    "e2e3",
    "e2e4",
    "e2e5",
    "e2e6",
    "e2e7",
    "e2e8",
    "e2d1",
    "e2f3",
    "e2g4",
    "e2h5",
    "e2a6",
    "e2b5",
    "e2c4",
    "e2d3",
    "e2f1",
    "e2c1",
    "e2c3",
    "e2g1",
    "e2d4",
    "e2g3",
    "e2f4",
    "e3a3",
    "e3b3",
    "e3c3",
    "e3d3",
    "e3f3",
    "e3g3",
    "e3h3",
    "e3e1",
    "e3e2",
    "e3e4",
    "e3e5",
    "e3e6",
    "e3e7",
    "e3e8",
    "e3c1",
    "e3d2",
    "e3f4",
    "e3g5",
    "e3h6",
    "e3a7",
    "e3b6",
    "e3c5",
    "e3d4",
    "e3f2",
    "e3g1",
    "e3c2",
    "e3d1",
    "e3c4",
    "e3f1",
    "e3g2",
    "e3d5",
    "e3g4",
    "e3f5",
    "e4a4",
    "e4b4",
    "e4c4",
    "e4d4",
    "e4f4",
    "e4g4",
    "e4h4",
    "e4e1",
    "e4e2",
    "e4e3",
    "e4e5",
    "e4e6",
    "e4e7",
    "e4e8",
    "e4b1",
    "e4c2",
    "e4d3",
    "e4f5",
    "e4g6",
    "e4h7",
    "e4a8",
    "e4b7",
    "e4c6",
    "e4d5",
    "e4f3",
    "e4g2",
    "e4h1",
    "e4c3",
    "e4d2",
    "e4c5",
    "e4f2",
    "e4g3",
    "e4d6",
    "e4g5",
    "e4f6",
    "e5a5",
    "e5b5",
    "e5c5",
    "e5d5",
    "e5f5",
    "e5g5",
    "e5h5",
    "e5e1",
    "e5e2",
    "e5e3",
    "e5e4",
    "e5e6",
    "e5e7",
    "e5e8",
    "e5a1",
    "e5b2",
    "e5c3",
    "e5d4",
    "e5f6",
    "e5g7",
    "e5h8",
    "e5b8",
    "e5c7",
    "e5d6",
    "e5f4",
    "e5g3",
    "e5h2",
    "e5c4",
    "e5d3",
    "e5c6",
    "e5f3",
    "e5g4",
    "e5d7",
    "e5g6",
    "e5f7",
    "e6a6",
    "e6b6",
    "e6c6",
    "e6d6",
    "e6f6",
    "e6g6",
    "e6h6",
    "e6e1",
    "e6e2",
    "e6e3",
    "e6e4",
    "e6e5",
    "e6e7",
    "e6e8",
    "e6a2",
    "e6b3",
    "e6c4",
    "e6d5",
    "e6f7",
    "e6g8",
    "e6c8",
    "e6d7",
    "e6f5",
    "e6g4",
    "e6h3",
    "e6c5",
    "e6d4",
    "e6c7",
    "e6f4",
    "e6g5",
    "e6d8",
    "e6g7",
    "e6f8",
    "e7a7",
    "e7b7",
    "e7c7",
    "e7d7",
    "e7f7",
    "e7g7",
    "e7h7",
    "e7e1",
    "e7e2",
    "e7e3",
    "e7e4",
    "e7e5",
    "e7e6",
    "e7e8",
    "e7a3",
    "e7b4",
    "e7c5",
    "e7d6",
    "e7f8",
    "e7d8",
    "e7f6",
    "e7g5",
    "e7h4",
    "e7c6",
    "e7d5",
    "e7c8",
    "e7f5",
    "e7g6",
    "e7g8",
    "e8a8",
    "e8b8",
    "e8c8",
    "e8d8",
    "e8f8",
    "e8g8",
    "e8h8",
    "e8e1",
    "e8e2",
    "e8e3",
    "e8e4",
    "e8e5",
    "e8e6",
    "e8e7",
    "e8a4",
    "e8b5",
    "e8c6",
    "e8d7",
    "e8f7",
    "e8g6",
    "e8h5",
    "e8c7",
    "e8d6",
    "e8f6",
    "e8g7",
    "f1a1",
    "f1b1",
    "f1c1",
    "f1d1",
    "f1e1",
    "f1g1",
    "f1h1",
    "f1f2",
    "f1f3",
    "f1f4",
    "f1f5",
    "f1f6",
    "f1f7",
    "f1f8",
    "f1g2",
    "f1h3",
    "f1a6",
    "f1b5",
    "f1c4",
    "f1d3",
    "f1e2",
    "f1d2",
    "f1e3",
    "f1h2",
    "f1g3",
    "f2a2",
    "f2b2",
    "f2c2",
    "f2d2",
    "f2e2",
    "f2g2",
    "f2h2",
    "f2f1",
    "f2f3",
    "f2f4",
    "f2f5",
    "f2f6",
    "f2f7",
    "f2f8",
    "f2e1",
    "f2g3",
    "f2h4",
    "f2a7",
    "f2b6",
    "f2c5",
    "f2d4",
    "f2e3",
    "f2g1",
    "f2d1",
    "f2d3",
    "f2h1",
    "f2e4",
    "f2h3",
    "f2g4",
    "f3a3",
    "f3b3",
    "f3c3",
    "f3d3",
    "f3e3",
    "f3g3",
    "f3h3",
    "f3f1",
    "f3f2",
    "f3f4",
    "f3f5",
    "f3f6",
    "f3f7",
    "f3f8",
    "f3d1",
    "f3e2",
    "f3g4",
    "f3h5",
    "f3a8",
    "f3b7",
    "f3c6",
    "f3d5",
    "f3e4",
    "f3g2",
    "f3h1",
    "f3d2",
    "f3e1",
    "f3d4",
    "f3g1",
    "f3h2",
    "f3e5",
    "f3h4",
    "f3g5",
    "f4a4",
    "f4b4",
    "f4c4",
    "f4d4",
    "f4e4",
    "f4g4",
    "f4h4",
    "f4f1",
    "f4f2",
    "f4f3",
    "f4f5",
    "f4f6",
    "f4f7",
    "f4f8",
    "f4c1",
    "f4d2",
    "f4e3",
    "f4g5",
    "f4h6",
    "f4b8",
    "f4c7",
    "f4d6",
    "f4e5",
    "f4g3",
    "f4h2",
    "f4d3",
    "f4e2",
    "f4d5",
    "f4g2",
    "f4h3",
    "f4e6",
    "f4h5",
    "f4g6",
    "f5a5",
    "f5b5",
    "f5c5",
    "f5d5",
    "f5e5",
    "f5g5",
    "f5h5",
    "f5f1",
    "f5f2",
    "f5f3",
    "f5f4",
    "f5f6",
    "f5f7",
    "f5f8",
    "f5b1",
    "f5c2",
    "f5d3",
    "f5e4",
    "f5g6",
    "f5h7",
    "f5c8",
    "f5d7",
    "f5e6",
    "f5g4",
    "f5h3",
    "f5d4",
    "f5e3",
    "f5d6",
    "f5g3",
    "f5h4",
    "f5e7",
    "f5h6",
    "f5g7",
    "f6a6",
    "f6b6",
    "f6c6",
    "f6d6",
    "f6e6",
    "f6g6",
    "f6h6",
    "f6f1",
    "f6f2",
    "f6f3",
    "f6f4",
    "f6f5",
    "f6f7",
    "f6f8",
    "f6a1",
    "f6b2",
    "f6c3",
    "f6d4",
    "f6e5",
    "f6g7",
    "f6h8",
    "f6d8",
    "f6e7",
    "f6g5",
    "f6h4",
    "f6d5",
    "f6e4",
    "f6d7",
    "f6g4",
    "f6h5",
    "f6e8",
    "f6h7",
    "f6g8",
    "f7a7",
    "f7b7",
    "f7c7",
    "f7d7",
    "f7e7",
    "f7g7",
    "f7h7",
    "f7f1",
    "f7f2",
    "f7f3",
    "f7f4",
    "f7f5",
    "f7f6",
    "f7f8",
    "f7a2",
    "f7b3",
    "f7c4",
    "f7d5",
    "f7e6",
    "f7g8",
    "f7e8",
    "f7g6",
    "f7h5",
    "f7d6",
    "f7e5",
    "f7d8",
    "f7g5",
    "f7h6",
    "f7h8",
    "f8a8",
    "f8b8",
    "f8c8",
    "f8d8",
    "f8e8",
    "f8g8",
    "f8h8",
    "f8f1",
    "f8f2",
    "f8f3",
    "f8f4",
    "f8f5",
    "f8f6",
    "f8f7",
    "f8a3",
    "f8b4",
    "f8c5",
    "f8d6",
    "f8e7",
    "f8g7",
    "f8h6",
    "f8d7",
    "f8e6",
    "f8g6",
    "f8h7",
    "g1a1",
    "g1b1",
    "g1c1",
    "g1d1",
    "g1e1",
    "g1f1",
    "g1h1",
    "g1g2",
    "g1g3",
    "g1g4",
    "g1g5",
    "g1g6",
    "g1g7",
    "g1g8",
    "g1h2",
    "g1a7",
    "g1b6",
    "g1c5",
    "g1d4",
    "g1e3",
    "g1f2",
    "g1e2",
    "g1f3",
    "g1h3",
    "g2a2",
    "g2b2",
    "g2c2",
    "g2d2",
    "g2e2",
    "g2f2",
    "g2h2",
    "g2g1",
    "g2g3",
    "g2g4",
    "g2g5",
    "g2g6",
    "g2g7",
    "g2g8",
    "g2f1",
    "g2h3",
    "g2a8",
    "g2b7",
    "g2c6",
    "g2d5",
    "g2e4",
    "g2f3",
    "g2h1",
    "g2e1",
    "g2e3",
    "g2f4",
    "g2h4",
    "g3a3",
    "g3b3",
    "g3c3",
    "g3d3",
    "g3e3",
    "g3f3",
    "g3h3",
    "g3g1",
    "g3g2",
    "g3g4",
    "g3g5",
    "g3g6",
    "g3g7",
    "g3g8",
    "g3e1",
    "g3f2",
    "g3h4",
    "g3b8",
    "g3c7",
    "g3d6",
    "g3e5",
    "g3f4",
    "g3h2",
    "g3e2",
    "g3f1",
    "g3e4",
    "g3h1",
    "g3f5",
    "g3h5",
    "g4a4",
    "g4b4",
    "g4c4",
    "g4d4",
    "g4e4",
    "g4f4",
    "g4h4",
    "g4g1",
    "g4g2",
    "g4g3",
    "g4g5",
    "g4g6",
    "g4g7",
    "g4g8",
    "g4d1",
    "g4e2",
    "g4f3",
    "g4h5",
    "g4c8",
    "g4d7",
    "g4e6",
    "g4f5",
    "g4h3",
    "g4e3",
    "g4f2",
    "g4e5",
    "g4h2",
    "g4f6",
    "g4h6",
    "g5a5",
    "g5b5",
    "g5c5",
    "g5d5",
    "g5e5",
    "g5f5",
    "g5h5",
    "g5g1",
    "g5g2",
    "g5g3",
    "g5g4",
    "g5g6",
    "g5g7",
    "g5g8",
    "g5c1",
    "g5d2",
    "g5e3",
    "g5f4",
    "g5h6",
    "g5d8",
    "g5e7",
    "g5f6",
    "g5h4",
    "g5e4",
    "g5f3",
    "g5e6",
    "g5h3",
    "g5f7",
    "g5h7",
    "g6a6",
    "g6b6",
    "g6c6",
    "g6d6",
    "g6e6",
    "g6f6",
    "g6h6",
    "g6g1",
    "g6g2",
    "g6g3",
    "g6g4",
    "g6g5",
    "g6g7",
    "g6g8",
    "g6b1",
    "g6c2",
    "g6d3",
    "g6e4",
    "g6f5",
    "g6h7",
    "g6e8",
    "g6f7",
    "g6h5",
    "g6e5",
    "g6f4",
    "g6e7",
    "g6h4",
    "g6f8",
    "g6h8",
    "g7a7",
    "g7b7",
    "g7c7",
    "g7d7",
    "g7e7",
    "g7f7",
    "g7h7",
    "g7g1",
    "g7g2",
    "g7g3",
    "g7g4",
    "g7g5",
    "g7g6",
    "g7g8",
    "g7a1",
    "g7b2",
    "g7c3",
    "g7d4",
    "g7e5",
    "g7f6",
    "g7h8",
    "g7f8",
    "g7h6",
    "g7e6",
    "g7f5",
    "g7e8",
    "g7h5",
    "g8a8",
    "g8b8",
    "g8c8",
    "g8d8",
    "g8e8",
    "g8f8",
    "g8h8",
    "g8g1",
    "g8g2",
    "g8g3",
    "g8g4",
    "g8g5",
    "g8g6",
    "g8g7",
    "g8a2",
    "g8b3",
    "g8c4",
    "g8d5",
    "g8e6",
    "g8f7",
    "g8h7",
    "g8e7",
    "g8f6",
    "g8h6",
    "h1a1",
    "h1b1",
    "h1c1",
    "h1d1",
    "h1e1",
    "h1f1",
    "h1g1",
    "h1h2",
    "h1h3",
    "h1h4",
    "h1h5",
    "h1h6",
    "h1h7",
    "h1h8",
    "h1a8",
    "h1b7",
    "h1c6",
    "h1d5",
    "h1e4",
    "h1f3",
    "h1g2",
    "h1f2",
    "h1g3",
    "h2a2",
    "h2b2",
    "h2c2",
    "h2d2",
    "h2e2",
    "h2f2",
    "h2g2",
    "h2h1",
    "h2h3",
    "h2h4",
    "h2h5",
    "h2h6",
    "h2h7",
    "h2h8",
    "h2g1",
    "h2b8",
    "h2c7",
    "h2d6",
    "h2e5",
    "h2f4",
    "h2g3",
    "h2f1",
    "h2f3",
    "h2g4",
    "h3a3",
    "h3b3",
    "h3c3",
    "h3d3",
    "h3e3",
    "h3f3",
    "h3g3",
    "h3h1",
    "h3h2",
    "h3h4",
    "h3h5",
    "h3h6",
    "h3h7",
    "h3h8",
    "h3f1",
    "h3g2",
    "h3c8",
    "h3d7",
    "h3e6",
    "h3f5",
    "h3g4",
    "h3f2",
    "h3g1",
    "h3f4",
    "h3g5",
    "h4a4",
    "h4b4",
    "h4c4",
    "h4d4",
    "h4e4",
    "h4f4",
    "h4g4",
    "h4h1",
    "h4h2",
    "h4h3",
    "h4h5",
    "h4h6",
    "h4h7",
    "h4h8",
    "h4e1",
    "h4f2",
    "h4g3",
    "h4d8",
    "h4e7",
    "h4f6",
    "h4g5",
    "h4f3",
    "h4g2",
    "h4f5",
    "h4g6",
    "h5a5",
    "h5b5",
    "h5c5",
    "h5d5",
    "h5e5",
    "h5f5",
    "h5g5",
    "h5h1",
    "h5h2",
    "h5h3",
    "h5h4",
    "h5h6",
    "h5h7",
    "h5h8",
    "h5d1",
    "h5e2",
    "h5f3",
    "h5g4",
    "h5e8",
    "h5f7",
    "h5g6",
    "h5f4",
    "h5g3",
    "h5f6",
    "h5g7",
    "h6a6",
    "h6b6",
    "h6c6",
    "h6d6",
    "h6e6",
    "h6f6",
    "h6g6",
    "h6h1",
    "h6h2",
    "h6h3",
    "h6h4",
    "h6h5",
    "h6h7",
    "h6h8",
    "h6c1",
    "h6d2",
    "h6e3",
    "h6f4",
    "h6g5",
    "h6f8",
    "h6g7",
    "h6f5",
    "h6g4",
    "h6f7",
    "h6g8",
    "h7a7",
    "h7b7",
    "h7c7",
    "h7d7",
    "h7e7",
    "h7f7",
    "h7g7",
    "h7h1",
    "h7h2",
    "h7h3",
    "h7h4",
    "h7h5",
    "h7h6",
    "h7h8",
    "h7b1",
    "h7c2",
    "h7d3",
    "h7e4",
    "h7f5",
    "h7g6",
    "h7g8",
    "h7f6",
    "h7g5",
    "h7f8",
    "h8a8",
    "h8b8",
    "h8c8",
    "h8d8",
    "h8e8",
    "h8f8",
    "h8g8",
    "h8h1",
    "h8h2",
    "h8h3",
    "h8h4",
    "h8h5",
    "h8h6",
    "h8h7",
    "h8a1",
    "h8b2",
    "h8c3",
    "h8d4",
    "h8e5",
    "h8f6",
    "h8g7",
    "h8f7",
    "h8g6",
    "a2a1q",
    "a7a8q",
    "a2b1q",
    "a7b8q",
    "a2a1r",
    "a7a8r",
    "a2b1r",
    "a7b8r",
    "a2a1b",
    "a7a8b",
    "a2b1b",
    "a7b8b",
    "a2a1n",
    "a7a8n",
    "a2b1n",
    "a7b8n",
    "b2b1q",
    "b7b8q",
    "b2a1q",
    "b7a8q",
    "b2c1q",
    "b7c8q",
    "b2b1r",
    "b7b8r",
    "b2a1r",
    "b7a8r",
    "b2c1r",
    "b7c8r",
    "b2b1b",
    "b7b8b",
    "b2a1b",
    "b7a8b",
    "b2c1b",
    "b7c8b",
    "b2b1n",
    "b7b8n",
    "b2a1n",
    "b7a8n",
    "b2c1n",
    "b7c8n",
    "c2c1q",
    "c7c8q",
    "c2b1q",
    "c7b8q",
    "c2d1q",
    "c7d8q",
    "c2c1r",
    "c7c8r",
    "c2b1r",
    "c7b8r",
    "c2d1r",
    "c7d8r",
    "c2c1b",
    "c7c8b",
    "c2b1b",
    "c7b8b",
    "c2d1b",
    "c7d8b",
    "c2c1n",
    "c7c8n",
    "c2b1n",
    "c7b8n",
    "c2d1n",
    "c7d8n",
    "d2d1q",
    "d7d8q",
    "d2c1q",
    "d7c8q",
    "d2e1q",
    "d7e8q",
    "d2d1r",
    "d7d8r",
    "d2c1r",
    "d7c8r",
    "d2e1r",
    "d7e8r",
    "d2d1b",
    "d7d8b",
    "d2c1b",
    "d7c8b",
    "d2e1b",
    "d7e8b",
    "d2d1n",
    "d7d8n",
    "d2c1n",
    "d7c8n",
    "d2e1n",
    "d7e8n",
    "e2e1q",
    "e7e8q",
    "e2d1q",
    "e7d8q",
    "e2f1q",
    "e7f8q",
    "e2e1r",
    "e7e8r",
    "e2d1r",
    "e7d8r",
    "e2f1r",
    "e7f8r",
    "e2e1b",
    "e7e8b",
    "e2d1b",
    "e7d8b",
    "e2f1b",
    "e7f8b",
    "e2e1n",
    "e7e8n",
    "e2d1n",
    "e7d8n",
    "e2f1n",
    "e7f8n",
    "f2f1q",
    "f7f8q",
    "f2e1q",
    "f7e8q",
    "f2g1q",
    "f7g8q",
    "f2f1r",
    "f7f8r",
    "f2e1r",
    "f7e8r",
    "f2g1r",
    "f7g8r",
    "f2f1b",
    "f7f8b",
    "f2e1b",
    "f7e8b",
    "f2g1b",
    "f7g8b",
    "f2f1n",
    "f7f8n",
    "f2e1n",
    "f7e8n",
    "f2g1n",
    "f7g8n",
    "g2g1q",
    "g7g8q",
    "g2f1q",
    "g7f8q",
    "g2h1q",
    "g7h8q",
    "g2g1r",
    "g7g8r",
    "g2f1r",
    "g7f8r",
    "g2h1r",
    "g7h8r",
    "g2g1b",
    "g7g8b",
    "g2f1b",
    "g7f8b",
    "g2h1b",
    "g7h8b",
    "g2g1n",
    "g7g8n",
    "g2f1n",
    "g7f8n",
    "g2h1n",
    "g7h8n",
    "h2h1q",
    "h7h8q",
    "h2g1q",
    "h7g8q",
    "h2h1r",
    "h7h8r",
    "h2g1r",
    "h7g8r",
    "h2h1b",
    "h7h8b",
    "h2g1b",
    "h7g8b",
    "h2h1n",
    "h7h8n",
    "h2g1n",
    "h7g8n",
    "N@a1",
    "B@a1",
    "R@a1",
    "Q@a1",
    "P@a2",
    "N@a2",
    "B@a2",
    "R@a2",
    "Q@a2",
    "P@a3",
    "N@a3",
    "B@a3",
    "R@a3",
    "Q@a3",
    "P@a4",
    "N@a4",
    "B@a4",
    "R@a4",
    "Q@a4",
    "P@a5",
    "N@a5",
    "B@a5",
    "R@a5",
    "Q@a5",
    "P@a6",
    "N@a6",
    "B@a6",
    "R@a6",
    "Q@a6",
    "P@a7",
    "N@a7",
    "B@a7",
    "R@a7",
    "Q@a7",
    "N@a8",
    "B@a8",
    "R@a8",
    "Q@a8",
    "N@b1",
    "B@b1",
    "R@b1",
    "Q@b1",
    "P@b2",
    "N@b2",
    "B@b2",
    "R@b2",
    "Q@b2",
    "P@b3",
    "N@b3",
    "B@b3",
    "R@b3",
    "Q@b3",
    "P@b4",
    "N@b4",
    "B@b4",
    "R@b4",
    "Q@b4",
    "P@b5",
    "N@b5",
    "B@b5",
    "R@b5",
    "Q@b5",
    "P@b6",
    "N@b6",
    "B@b6",
    "R@b6",
    "Q@b6",
    "P@b7",
    "N@b7",
    "B@b7",
    "R@b7",
    "Q@b7",
    "N@b8",
    "B@b8",
    "R@b8",
    "Q@b8",
    "N@c1",
    "B@c1",
    "R@c1",
    "Q@c1",
    "P@c2",
    "N@c2",
    "B@c2",
    "R@c2",
    "Q@c2",
    "P@c3",
    "N@c3",
    "B@c3",
    "R@c3",
    "Q@c3",
    "P@c4",
    "N@c4",
    "B@c4",
    "R@c4",
    "Q@c4",
    "P@c5",
    "N@c5",
    "B@c5",
    "R@c5",
    "Q@c5",
    "P@c6",
    "N@c6",
    "B@c6",
    "R@c6",
    "Q@c6",
    "P@c7",
    "N@c7",
    "B@c7",
    "R@c7",
    "Q@c7",
    "N@c8",
    "B@c8",
    "R@c8",
    "Q@c8",
    "N@d1",
    "B@d1",
    "R@d1",
    "Q@d1",
    "P@d2",
    "N@d2",
    "B@d2",
    "R@d2",
    "Q@d2",
    "P@d3",
    "N@d3",
    "B@d3",
    "R@d3",
    "Q@d3",
    "P@d4",
    "N@d4",
    "B@d4",
    "R@d4",
    "Q@d4",
    "P@d5",
    "N@d5",
    "B@d5",
    "R@d5",
    "Q@d5",
    "P@d6",
    "N@d6",
    "B@d6",
    "R@d6",
    "Q@d6",
    "P@d7",
    "N@d7",
    "B@d7",
    "R@d7",
    "Q@d7",
    "N@d8",
    "B@d8",
    "R@d8",
    "Q@d8",
    "N@e1",
    "B@e1",
    "R@e1",
    "Q@e1",
    "P@e2",
    "N@e2",
    "B@e2",
    "R@e2",
    "Q@e2",
    "P@e3",
    "N@e3",
    "B@e3",
    "R@e3",
    "Q@e3",
    "P@e4",
    "N@e4",
    "B@e4",
    "R@e4",
    "Q@e4",
    "P@e5",
    "N@e5",
    "B@e5",
    "R@e5",
    "Q@e5",
    "P@e6",
    "N@e6",
    "B@e6",
    "R@e6",
    "Q@e6",
    "P@e7",
    "N@e7",
    "B@e7",
    "R@e7",
    "Q@e7",
    "N@e8",
    "B@e8",
    "R@e8",
    "Q@e8",
    "N@f1",
    "B@f1",
    "R@f1",
    "Q@f1",
    "P@f2",
    "N@f2",
    "B@f2",
    "R@f2",
    "Q@f2",
    "P@f3",
    "N@f3",
    "B@f3",
    "R@f3",
    "Q@f3",
    "P@f4",
    "N@f4",
    "B@f4",
    "R@f4",
    "Q@f4",
    "P@f5",
    "N@f5",
    "B@f5",
    "R@f5",
    "Q@f5",
    "P@f6",
    "N@f6",
    "B@f6",
    "R@f6",
    "Q@f6",
    "P@f7",
    "N@f7",
    "B@f7",
    "R@f7",
    "Q@f7",
    "N@f8",
    "B@f8",
    "R@f8",
    "Q@f8",
    "N@g1",
    "B@g1",
    "R@g1",
    "Q@g1",
    "P@g2",
    "N@g2",
    "B@g2",
    "R@g2",
    "Q@g2",
    "P@g3",
    "N@g3",
    "B@g3",
    "R@g3",
    "Q@g3",
    "P@g4",
    "N@g4",
    "B@g4",
    "R@g4",
    "Q@g4",
    "P@g5",
    "N@g5",
    "B@g5",
    "R@g5",
    "Q@g5",
    "P@g6",
    "N@g6",
    "B@g6",
    "R@g6",
    "Q@g6",
    "P@g7",
    "N@g7",
    "B@g7",
    "R@g7",
    "Q@g7",
    "N@g8",
    "B@g8",
    "R@g8",
    "Q@g8",
    "N@h1",
    "B@h1",
    "R@h1",
    "Q@h1",
    "P@h2",
    "N@h2",
    "B@h2",
    "R@h2",
    "Q@h2",
    "P@h3",
    "N@h3",
    "B@h3",
    "R@h3",
    "Q@h3",
    "P@h4",
    "N@h4",
    "B@h4",
    "R@h4",
    "Q@h4",
    "P@h5",
    "N@h5",
    "B@h5",
    "R@h5",
    "Q@h5",
    "P@h6",
    "N@h6",
    "B@h6",
    "R@h6",
    "Q@h6",
    "P@h7",
    "N@h7",
    "B@h7",
    "R@h7",
    "Q@h7",
    "N@h8",
    "B@h8",
    "R@h8",
    "Q@h8",
};

// legal moves total which are represented in the NN
const int NB_LABELS = 2272;

// will be filled in init()
//static std::vector <std::string> LABELS_MIRRORED;

// stores a mapping from Stockfish's move representation to the NN index in the policy
extern std::unordered_map<Move, size_t> MV_LOOKUP; // = {};
extern std::unordered_map<Move, size_t> MV_LOOKUP_MIRRORED; // = {};

//https://stackoverflow.com/questions/23390034/c-change-global-variable-value-in-different-files
extern std::string LABELS_MIRRORED[NB_LABELS];

/**
 * @brief mirror_move Mirrors an move in uci representation
 * @param moveUCI String uci
 * @return Returns corresponding mirrored uci string with the rank flipped
 */
std::string mirror_move(std::string moveUCI);

namespace Constants {
//void init();
inline void init() {

    // fill mirrored label list and look-up table
    for (size_t mvIdx=0; mvIdx < NB_LABELS; mvIdx++) {
//        if (mvIdx == 2069)
//        std::cout << "mvIdx " << mvIdx << "mirror_move(LABELS[mvIdx])" << mirror_move(LABELS[mvIdx]) << std::endl;
//        std::cout << "mvIdx" << mvIdx << std::endl;
        LABELS_MIRRORED[mvIdx] = mirror_move(LABELS[mvIdx]);
//        LABELS_MIRRORED.push_back(std::string("test")); //mirror_move(LABELS[mvIdx])));
        std::vector<Move> moves = make_move(LABELS[mvIdx]);
        for (Move move : moves) {
            MV_LOOKUP.insert({move, mvIdx});
//            std::cout << "insert " << move << " " << mvIdx << std::endl;
        }
//        std::cout << LABELS_MIRRORED[mvIdx] << std::endl;
        std::vector<Move> moves_mirrored = make_move(LABELS_MIRRORED[mvIdx]);
        for (Move move : moves_mirrored) {
            MV_LOOKUP_MIRRORED.insert({move, mvIdx});
        }
    }
}
}

/*
def mirror_move(move: chess.Move):
    """
    Mirrors a move given as python chess notation

    :param move: Move object
    :return: Mirrored move
    """
    from_square = chess.square_mirror(move.from_square)
    to_square = chess.square_mirror(move.to_square)
    return chess.Move(from_square, to_square, move.promotion, move.drop)


# flip the labels for BLACK
LABELS_MIRRORED = [None] * NB_LABELS

for i, label in enumerate(LABELS):
    mv = chess.Move.from_uci(label)
    mv_mirrored = mirror_move(mv)
    LABELS_MIRRORED[i] = mv_mirrored.uci()


# The movement lookup table is a dictionary/hash-map which maps the string to the corresponding label index
MV_LOOKUP = {}

# iterate over all moves and assign the integer move index to the string
for i, label in enumerate(LABELS):
    MV_LOOKUP[label] = i


# do the same for the black player
MV_LOOKUP_MIRRORED = {}

# iterate over all moves and assign the integer move index to the string
for i, label in enumerate(LABELS_MIRRORED):
    MV_LOOKUP_MIRRORED[label] = i
*/

#endif // CONSTANTS_H
