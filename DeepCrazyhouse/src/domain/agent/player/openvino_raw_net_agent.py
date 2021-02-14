#!/usr/bin/env python3
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from time import time
from DeepCrazyhouse.src.domain.abstract_cls.abs_game_state import AbsGameState
from DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
from DeepCrazyhouse.src.domain.abstract_cls.abs_agent import AbsAgent
from DeepCrazyhouse.src.domain.variants.output_representation import get_probs_of_move_list, value_to_centipawn

from openvino.inference_engine import IECore


class OpenvinoRawNetAgent(AbsAgent):
    """ Builds the raw network"""

    def __init__(self, net: NeuralNetAPI, temperature=0.0, temperature_moves=4, verbose=True):
        super().__init__(temperature, temperature_moves, verbose)
        self._net = net

    def evaluate_board_state(self, state: AbsGameState):  # Too few public methods (1/2)
        """
        The greedy agent always performs the first legal move with the highest move probability

        :param state: Gamestate object
        :return:
        value - Value prediction in the current players view from [-1,1]: -1 -> 100% lost, +1 100% won
        selected_move - Python chess move object of the selected move
        confidence - Probability value for the selected move in the probability distribution
        idx - Integer index of the move which was returned
        centipawn - Centi pawn evaluation which is converted from the value prediction in currents player view
        depth - Depth which was reached after the search
        nodes - Number of nodes which have been evaluated in the search
        time_elapsed_s - Elapsed time in seconds for the full search
        nps - Nodes per second metric
        pv - Calculated best line for both players
        """

        t_start_eval = time()

        # Start sync inference
        print("Starting inference")

        print("Preparing input blobs")
        input_blob = next(iter(self._net.read_net.input_info))
        output_blob=iter(self._net.read_net.outputs)
        pred_policy_blob = next(output_blob)
        pred_value_blob = next(output_blob)

        # NB: This is required to load the image as uint8 np.array
        #     Without this step the input blob is loaded in FP32 precision,
        #     this requires additional operation and more memory.
        self._net.read_net.input_info[input_blob].precision = "U8"

        res = self._net.exec_net.infer(inputs={input_blob: state.get_state_planes()})
        
        #TODO Check order of output
        
        pred_value=res[pred_value_blob][0][0]
        pred_policy=res[pred_policy_blob][0]

        legal_moves = list(state.get_legal_moves())
        p_vec_small = get_probs_of_move_list(pred_policy, legal_moves, state.is_white_to_move())
        # define the remaining return variables
        time_e = time() - t_start_eval
        centipawn = value_to_centipawn(pred_value)
        depth = nodes = 1
        time_elapsed_s = time_e * 1000
        nps = nodes / time_e
        # use the move with the highest probability as the best move for logging
        pv = legal_moves[p_vec_small.argmax()].uci()
        return pred_value, legal_moves, p_vec_small, centipawn, depth, nodes, time_elapsed_s, nps, pv









