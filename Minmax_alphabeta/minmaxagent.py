from exceptions import AgentException
from connect4 import Connect4
from copy import deepcopy
from random import random

class MinMaxAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4: Connect4, depth = 5, showAdvantage = True):
        #maximizing the agent for first move
        value =  float('-inf')
        move = None
        for possible_drop in connect4.possible_drops():
            mygame = deepcopy(connect4)
            mygame.drop_token(possible_drop)
            new_value = self.minmax(mygame, depth - 1)
            if new_value > value:
                value = new_value
                move = possible_drop
            #adding some non-determinism to the algorithm, so that it's less predictable
            #could also add a tolerance in which the alghoritm would have a chace to pick a worse move
            elif new_value == value and random()<0.5:
                value = new_value
                move = possible_drop

        if showAdvantage:
            print(f'min-max advantage <-1, 1> {value}')
        return move
        
        
    def minmax(self, connect4: Connect4, depth):
        #max depth reached or terminal node
        if depth == 0 or connect4._check_game_over():
            return self.heuristicGrade(connect4)
        
        #minimizing agent
        if connect4.who_moves != self.my_token:
            value =  float('inf')
            for possible_drop in connect4.possible_drops():
                mygame = deepcopy(connect4)
                mygame.drop_token(possible_drop)
                new_value = self.minmax(mygame, depth - 1)
                if new_value < value:
                    value = new_value
            return value

        #maximizing agent
        else:
            value =  float('-inf')
            for possible_drop in connect4.possible_drops():
                mygame = deepcopy(connect4)
                mygame.drop_token(possible_drop)
                new_value = self.minmax(mygame, depth - 1)
                if new_value > value:
                    value = new_value
            return value

    
    def heuristicGrade(self, connect4: Connect4):
        #game over results
        if connect4._check_game_over():
            if connect4.wins == self.my_token:
                return 1
            elif connect4.wins == None:
                return 0
            else:
                return -1
            
        #initial value of node = 0 (neutral)
        #weight aims to keep the end value in (-1; 1), while using the range as much as possible
        value =  0
        weight = 0.1 / connect4.height / connect4.width

        #agent's points
        for four in connect4.iter_fours():
            value += self.countPoints(four, self.my_token, weight)

        #opponent's points
        for four in connect4.iter_fours():
            value -= self.countPoints(four, 'x' if self.my_token=='o' else 'o', weight)

        if value>=1 or value<=-1:
            raise AgentException(f'Heuristic grade out of permitted range: {value}')
        
        return value
        

    def countPoints(self, four: list, token, weight):
        #counting the amount of our tokens in a given four
        #if we encounter an opponent's token, returns 0
        value = 0
        for element in four:
            if element != token and element != '_':
                return 0
            if element == token:
                value += 1
        
        #bias towards more completed, capped by weight to stay in the permitted range
        return value ** 2 * weight