# import gym
# import numpy as np
# from typing import List, Tuple

# class TakActions():
#     def __init__(self, board_size: int, stones: int) -> None:
#         self.board_size = board_size    
#         self.stones = stones
#         self.pieces = stones
#         self.max_pieces = stones * 2
#         self.carry_limit = board_size

#         actions = []
        
#         # Placing pieces into each space
#         for column in range(0, board_size):
#             for row in range(0, board_size):
#                 for type_ in ['Flat_Stone', 'Standing_Stone']:
#                     actions.append((2, column, row, type_))


#             for adjacent in self.get_adjacent_coordinates(column, row):
#                 for pieces in range(1, self.carry_limit + 1):
#                     actions.append((2, (column, row), adjacent, pieces))
                    
#     def get_adjacent_coordinates(self, column: int, row: int) -> List[Tuple[int, int]]:
#         adjacent = []

#         travel_range = list(range(-self.board_size+1,0)) + list(range(1,self.board_size))

#         for offset in travel_range:
#             new_column = column + offset
#             if 0 <= new_column < self.board_size:
#                 adjacent.append((new_column, row))

#             new_row = row + offset
#             if 0 <= new_row < self.board_size:
#                 adjacent.append((column, new_row))

#         return adjacent
