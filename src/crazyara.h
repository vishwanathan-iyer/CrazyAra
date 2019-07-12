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
 * @file: crazyara.h
 * Created on 12.06.2019
 * @author: queensgambit
 *
 * Main entry point for the executable which manages the UCI communication.
 */

#ifndef CRAZYARA_H
#define CRAZYARA_H

#include <iostream>

#include "agents/rawnetagent.h"
#include "agents/mctsagent.h"
#include "nn/neuralnetapi.h"
#include "agents/config/searchsettings.h"
#include "agents/config/searchlimits.h"
#include "agents/config/playsettings.h"
#include "node.h"
#include "statesmanager.h"


class CrazyAra
{
private:
    std::string intro = std::string("\n") +
            std::string("                                  _                                           \n") +
            std::string("                   _..           /   ._   _.  _        /\\   ._   _.           \n") +
            std::string("                 .' _ `\\         \\_  |   (_|  /_  \\/  /--\\  |   (_|           \n") +
            std::string("                /  /e)-,\\                         /                           \n") +
            std::string("               /  |  ,_ |                    __    __    __    __             \n") +
            std::string("              /   '-(-.)/          bw     8 /__////__////__////__////         \n") +
            std::string("            .'--.   \\  `                 7 ////__////__////__////__/          \n") +
            std::string("           /    `\\   |                  6 /__////__////__////__////           \n") +
            std::string("         /`       |  / /`\\.-.          5 ////__////__////__////__/            \n") +
            std::string("       .'        ;  /  \\_/__/         4 /__////__////__////__////             \n") +
            std::string("     .'`-'_     /_.'))).-` \\         3 ////__////__////__////__/              \n") +
            std::string("    / -'_.'---;`'-))).-'`\\_/        2 /__////__////__////__////               \n") +
            std::string("   (__.'/   /` .'`                 1 ////__////__////__////__/                \n") +
            std::string("    (_.'/ /` /`                       a  b  c  d  e  f  g  h                  \n") +
            std::string("      _|.' /`                                                                 \n") +
            std::string("jgs.-` __.'|  Developers: Johannes Czech, Moritz Willig, Alena Beyer          \n") +
            std::string("    .-'||  |  Source-Code: QueensGambit/CrazyAra (GPLv3-License)              \n") +
            std::string("       \\_`/   Inspiration: A0-paper by Silver, Hubert, Schrittwieser et al.   \n") +
            std::string("              ASCII-Art: Joan G. Stark, Chappell, Burton                      \n");
    RawNetAgent *rawAgent;
    MCTSAgent *mctsAgent;
    NeuralNetAPI *netSingle;
    NeuralNetAPI *netBatch;
    bool networkLoaded = false;
    StatesManager *states;

public:
    CrazyAra();
    void welcome();
    void uci_loop(int argc, char* argv[]);
    void init();

    /**
     * @brief is_ready Loads the neural network weights and creates the agent object in case there haven't loaded already
     * @return True, if everything isReady
     */
    bool is_ready();

    void go(Board *pos, istringstream& is);
    void position(Board *pos, istringstream &is);
};


int main(int argc, char* argv[]) {
    CrazyAra crazyara;
    crazyara.init();
    crazyara.welcome();
    crazyara.uci_loop(argc, argv);
}

// Challenging FENs
/*
NR1n1k1r/ppPbbppp/3p1b1n/8/4B3/2P5/P1P1pPPP/2n2R1K/QPQPr w - - 0 29
-> Q@e1
r1b1Rq1k/ppp2pqp/5Nn1/1B2p1B1/3P4/8/PPP2bpP/2KR2R1/PNPppn b - - 0 20
-> @h6 (Sf needs very high depth to find this move > ) (h7h6 is also viable) -> SF needs depth>= 19 to find this
r1b1kb1r/ppp1pppp/2n5/1B1qN3/3P2p1/2P5/P1P2PpP/R1BQK1R1/NPn w Qkq - 0 10
-> N@a6
3k2r1/pBpr1p1p/Pp3p1B/3p4/2PPn2B/5NPp/q4PpP/1R1QR1K1/NNbp w - - 1 23
-> N@e6
r1bq2nr/ppppbkpp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R[Pb] w KQ - 0 5
-> @d4
r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1[-] b kq - 0 4
-> d6?!

1r1k2r1/pP1bbpPp/5p2/3N4/4p3/8/PPnPQPPP/R1B2K1R/PBNqnpp b - - 1 21
-> N@c7
8/3pbkbP/N2pp1p1/1p2pprp/1P1P4/1Q6/P1P2P1P/2N1NK1R/BQprrbn b - - 0 57
-> P@e2 M#7
8/3pbkbP/N2pp1p1/1p2pprp/1PQP4/8/P1P2P1P/2N1NK1R/BBQprrn b - - 0 58
-> R@g1 M#11
r4rk1/ppp2pbp/8/4p1q1/3nB3/2NP1BPp/PPP2P1P/R4RK1[QBNPnp] b - - 0 21
-> N@d2 ?!
*/



#endif // CRAZYARA_H
