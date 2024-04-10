from connect4 import Connect4
from copy import deepcopy
from minmaxagent import MinMaxAgent
from random import random

#class derives from MinMaxAgent to inherit heuristicGrade and countPoints methods
class AlphaBetaAgent(MinMaxAgent):
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4: Connect4, depth = 5, showAdvantage = True):
        #maximizing the agent for first move
        value =  float('-inf')
        move = None
        alpha = float('-inf')
        beta = float('inf')
        for possible_drop in connect4.possible_drops():
            mygame = deepcopy(connect4)
            mygame.drop_token(possible_drop)
            new_value = self.alphabeta(mygame, depth - 1, alpha, beta)
            if new_value > value:
                value = new_value
                move = possible_drop
            #adding some non-determinism to the algorithm, so that it's less predictable
            #could also add a tolerance in which the alghoritm would have a chace to pick a worse move
            elif new_value == value and random()<0.5:
                value = new_value
                move = possible_drop

            if new_value > alpha:
                alpha = new_value
                
        if showAdvantage:
            print(f'alpha-beta advantage <-1, 1> {value}')
        return move
        
        
    def alphabeta(self, connect4: Connect4, depth, alpha, beta):
        #max depth reached or terminal node
        if depth == 0 or connect4._check_game_over():
            return self.heuristicGrade(connect4)
        
        #minimizing agent
        if connect4.who_moves != self.my_token:
            value =  float('inf')
            for possible_drop in connect4.possible_drops():
                mygame = deepcopy(connect4)
                mygame.drop_token(possible_drop)
                new_value = self.alphabeta(mygame, depth - 1, alpha, beta)
                if new_value < value:
                    value = new_value
                if new_value < beta:
                    beta = new_value
                if new_value <= alpha:
                    break
            return value

        #maximizing agent
        else:
            value =  float('-inf')
            for possible_drop in connect4.possible_drops():
                mygame = deepcopy(connect4)
                mygame.drop_token(possible_drop)
                new_value = self.alphabeta(mygame, depth - 1, alpha, beta)
                if new_value > value:
                    value = new_value
                if new_value > alpha:
                    alpha = new_value
                if new_value >= beta:
                    break
            return value