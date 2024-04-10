from exceptions import GameplayException
from connect4 import Connect4
from alphabetaagent import AlphaBetaAgent
from minmaxagent import MinMaxAgent
from randomagent import RandomAgent
from os import system

wins = [0, 0, 0]
n = 1000
for i in range(n):
    connect4 = Connect4()

    if i%2==0:
        agent1 = AlphaBetaAgent('o')
        agent2 = RandomAgent('x')
    else:
        agent1 = AlphaBetaAgent('x')
        agent2 = RandomAgent('o')

    while not connect4.game_over:
        #connect4.draw()
        try:
            if connect4.who_moves == agent1.my_token:
                n_column = agent1.decide(connect4, showAdvantage=False)
            else:
                n_column = agent2.decide(connect4)
            connect4.drop_token(n_column)
        except (ValueError, GameplayException):
            print('invalid move')

    #connect4.draw()
    if connect4.wins == agent1.my_token:
        wins[0]+=1
    elif connect4.wins == agent2.my_token:
        wins[1]+=1
    else:
        wins[2]+=1

    system('cls')
    print(f'Done {i+1}/{n}')
    

print(f'Wins alpha-beta:    {wins[0] / (wins[0]+wins[1]+wins[2]) * 100}%')
print(f'Wins random:        {wins[1] / (wins[0]+wins[1]+wins[2]) * 100}%')
print(f'Draws:              {wins[2] / (wins[0]+wins[1]+wins[2]) * 100}%')
