import gym
from gym import spaces
import pygame
import numpy as np
import copy
from typing import List, Tuple
from collections import defaultdict
import time

# Dictionary to store total time and call count for each function
func_stats = defaultdict(lambda: {'total_time': 0, 'call_count': 0})

def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        func_stats[func.__name__]['total_time'] += elapsed_time
        func_stats[func.__name__]['call_count'] += 1
        return result
    return wrapper

class TakEnv(gym.Env):
    """
    Tak OpenAI Gym Implementation
    Tak is modeled as a perfect information, Alternating Markov Game

    State
    - The state of the environment is a dictionary that contains the current board state and whose turn it is
    - The board is a square matrix the size of the board size
    - Each space on the board contains information about the pieces it contains
    - Initally the space is initialized as an empty list

    Example 3x3 Tak board
       [[], [], [],
       [], [], [],
       [], [], []]

    - When a piece is placed or moved to a tile, an array representing the stack of stones is added        
    
        - 0: Denotes an empty space
    
        - Player 1 Stones
        - 1: Denotes a flat stone from player 1
        - 3: Denotes a standing stone from player 1
        - 5: Denotes a capstone from player 1

        - Player 2 stones
        - 2: Denotes a flat stone from player 2
        - 4: Denotes a standing stone from player 2
        - 6: Denotes a capstone from player 2

        - The stack array represents the ordered stack of stones on each tile
       Example 3x3 tile stack
        [3, 1, 1, 2, 1]

        [3], - Top of stack, player 1 standing stone
        [1], - player 1 flat stone
        [1], - player 1 flat stone
        [2], - player 2 flat stone
        [1], - player 1 flat stone

    Actions
    - Fundamentally, there are two type of actions each player can take: Place a stone or move a stone/stack
    - Place Action
        - Place either a flat stone or a standing stone into an empty space on the board
    
    - Move Action
        - Move a stone or stack of stones the player contols to an adjacent space or spaces equal to the number of stones moved
        - A player controls a stack if the player controls the top stone
        - A player can only move a number of stones equal to the size of the board. (example: for a 3x3 board, a play can move at
            most 3 stones)

    - The action space is equal to every possible "place" action plus every possible "move" action as is represented as an integer
        in a discrete distribution the size of the total number of actions

    Reward
    - For a zero sum game, the reward is +1, -1, or 0 depending on if the current player wins, loses, or ties
    - A player wins if one of the following is true:
        - A road is created two adjacent sides of the board
            - A road is created if the top stone in a stack is a flat stone or capstone controlled be the player
            - A road does not need to be straight so long as a continous chain of stones connects one side of the board to the other
        
            Example 3x3 Board:
            [[list([]) list([]) list([2, 1, 1])]
            [list([2]) list([]) list([])]
            [list([1]) list([1]) list([1])]]
            Player 1 has a road connecting the left and right side of the board
        - If the board is filled, whoever controls the most flat stones on top of each stack wins the game
            - Capstones count as flat stones
            - If the players control equal amounts of flat stones, then the game is a tie
    
    """
    # Left over from other Tak implementations, human reden mode activates pygame
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, board_size, pieces, render_mode=None):
        self.board_size = board_size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        # TODO Game ends if a player runs out of pieces
        self.pieces = pieces

        # Initalize the board state
        self.state = self.new_board()

        # TODO Observation of the board space, aigym needs observations but because this is a fully observable game,
        # the observation will just be the state after taking an action plus any additional information that may need to be added
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Discrete(1)
            }
        )

        # Initialize the reward
        self.reward = 0

        # Discrete action space
        self.action_space = spaces.Discrete(len(self.takActions()))
        
        # Action represented as tuples
        self.actions = self.takActions()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

            self.clock = pygame.time.Clock()

            # Load piece images
            self.piece_images = {
                i: pygame.image.load(f"/Users/natefoote/DMU++/footetak/images/{i}.png") for i in range(1, 7)
            }
            # scale images
            pix_square_size = round(self.window_size / self.board_size * 0.66)
            for key, image in self.piece_images.items():
                # Get original dimensions
                orig_width, orig_height = image.get_size()
                scale_factor = min(
                    pix_square_size / orig_width, pix_square_size / orig_height
                )

                # Calculate new dimensions
                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)

                # Resize image
                self.piece_images[key] = pygame.transform.scale(
                    image, (new_width, new_height)
                )
        
        
        #self.window = None
        #self.clock = None

        # Reset the environment components to initial stats
        self.reset()

    def return_time(self):
        return func_stats

    # Environment must be reset before actions can be taken
    @time_tracker
    def reset(self, seed=None, options=None):

        self.state = self.new_board()

        observation = {'obs': self.state}
        info = {'info': 0}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    @time_tracker
    def is_terminal(self):
    # Check if game is done
        terminated = False
        game_over = False

        if self.pieces <= self.stone_count():
            game_over = True

        # If the current player creates a road, then collect reward
        if self.is_road(self.state['Player: ']):
            game_over = True

        # If the other player creates a road, then collect negative reward
        if self.is_road(-self.state['Player: ']):
            game_over = True

        # If the board is filled, award points to whoever controls the most flat stones
        if self.board_filled():
            game_over = True
        
        # Record the game as terminated
        if game_over == True:
            terminated = True
        
        return terminated
    
    @time_tracker
    def stone_count(self):
        count = 0
        for row in self.state["Board: "]:
            for column in row:
                count += len(column)
        
        return count
   
    @time_tracker
    def clone(self):
        return copy.deepcopy(self)

    
    # Step into the environment given a discrete action
    @time_tracker
    def step(self, action):

        terminated = False
        game_over = False
        reward = 0

        # Record observation
        observation = {'obs': 0}
        
        # Record additional information, likely not needed
        info = {'info': 0}
        
        #  Map the discrete action to a place or move action
        action = self.map_to_action(action)

        # Check if the move is valid
        if not self.valid_action(action):
            print("Action is not valid")
            terminated = True

            return observation, reward, terminated, info
        
        # Perform action
        if action[0] == 1:
            self.move_stone(action)
        else:
            self.place_stone(action)

        if self.is_terminal():
            terminated = True
            if self.is_road(self.state['Player: ']):
                reward = 1

            # If the other player creates a road, then collect negative reward
            elif self.is_road(-self.state['Player: ']):
                reward = -1

            else:
                player_1_points = len(np.argwhere(self.update_top_stones() == 1))
                player_2_points = len(np.argwhere(self.update_top_stones() == 2))
                if player_1_points == player_2_points:
                    reward = 0
                elif player_1_points > player_2_points:
                    if self.state['Player: '] == 1:
                        reward = 1
                    else:
                        reward = -1
                else:
                    if self.state['Player: '] == -1:
                        reward = 1
                    else:
                        reward = -1
         
        # Switch player
        if not terminated:
            self.state['Player: '] = - self.state['Player: ']

        # Render frame in pygame
        if self.render_mode == "human":
            self._render_frame()

        # Return a tuple containing:
        return observation, reward, terminated, info

    # Pygame functions
    @time_tracker
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    @time_tracker
    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (
            self.window_size / self.board_size
        )  # The size of a single grid square in pixels

        # Add gridlines
        for x in range(self.board_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # Draw towers
        for x in range(self.board_size):
            for y in range(self.board_size):
                tower = self.state["Board: "][x][
                    y
                ]  # Assuming self.board is the 2D array of towers
                base_x, base_y = (
                    x * pix_square_size + pix_square_size / 4,
                    y * pix_square_size + pix_square_size / 4,
                )
                offset = 0
                for piece in tower[::-1]:
                    img = self.piece_images[piece]
                    canvas.blit(img, (base_x, base_y - offset))
                    offset += 10  # Adjust this value as needed for proper visual offset

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    @time_tracker 
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

### Helper Functions ###
    
    # Create a clean board
    @time_tracker
    def new_board(self):

        # Create an NxN matrix of empty lists depending on the board size
        state_matrix = np.empty((self.board_size, self.board_size), dtype=list)
        state_matrix.fill([])

        # Return a dictionary containing the board and the current player
        return {"Board: ": state_matrix, "Player: ": 1}

    # Enumerate of all possible actions
    ''' 
    Actions are either to place a piece or move a stack of pieces
    Place actions are represented as a tuple in the following form:
    (identifier, (row, column), stone_type, discrete action)
        identifier - place (2) or move (1) action
        (row, column) - tuple containing row and column to place the stone on the board
        stone_type - type of stone being places: flat stone (1), standing stone (3), or capstone (5)
        discrete action - integer used to map discrete action space to action
        Ex 3x3 board place action: (2, (2,0), 1, 85) = place a flat stone in row 3, column 1

    Move actions are represented as a tuple in the following form:
    (identifier, (move_from_row, move_from_column), (move_to_row, move_to_column), (trail, ), discrete action)
        identifier - place (2) or move (1) action
        (move_from_row, move_from_column) - tuple containing row and column of stone or stones that are being moved from
        (move_to_row, move_to_column) - tuple containing the final row and column of stone or stones that are being moved to
        (trail, ) - tuple containing the number of stones being dropped in each space 
        discrete action - integer used to map discrete action space to action
        Ex 3x3 board move actions: 
        (1, (1, 2), (0, 2), (2,), 80) = move 2 stones from row 2, column 3 to row 1, column 3. Leave 2 stones in the first space of move
        (1, (0, 1), (2, 1), (2, 1), 27) = move 3 stones from row 1, column 2 to row 3, column 2. Leave 2 stones in the first space of move, and 1 stone in the second space of move
        
    Place actions are recorded from the perspective of player 1 meaning stone types are only 1,3 and 5. If player 2 places a stone, +1 is added to the stone type in the place_stone method
    Move actions are representitive of both players action space
    '''
    @time_tracker
    def takActions(self):   
        # Initalize an empty array for actions
        actions = []

        # Determine the max carry limit
        carry_limit = self.board_size

        # If the board size is greater than 5, include capstones (5 or 6 depending on the player)
        if self.board_size >= 5:
            types = [1, 3, 5]
        else:
            types = [1, 3]

        # Index for discrete moves
        i = 0

        # Enumerate through each row and column of the board
        for row in range(0, self.board_size):
            for column in range(0, self.board_size):

                # Place action for a stone of each type into each space
                for type_ in types:
                    i += 1
                    actions.append((2, (row, column), type_, i))

                # Enumerature through the possible move combinations
                # Enumerate through each space in the horizontal and vertical directions for a given space
                for adjacent in self.get_adjacent_coordinates(row, column):

                    # Enumerate through every possible trail combination of moving a stack of stones
                    for trail in self.generate_combinations(carry_limit,[]):

                        travel_distance = max(tuple(abs(x - y) for x, y in zip((row, column), adjacent)))

                        stacks_left_in_space = sum(1 for stack in trail if stack != 0)

                         # All stones are placed in the immediate adjactent space
                        if  travel_distance == stacks_left_in_space:
                            i = i + 1
                            actions.append((1, (row, column), adjacent, trail, i))

        # Return a list of actions        
        return actions

    # Get all the horizontal and vertical spaces adgacent to each space
    @time_tracker
    def get_adjacent_coordinates(self, row: int, column: int) -> List[Tuple[int, int]]:
        
        # Initalize empty list
        adjacent = []
        
        # Set the max travel range based on the board size
        travel_range = list(range(-self.board_size + 1,0)) + list(range(1, self.board_size))

        # Enumerate through each space that is within the board space
        for offset in travel_range:
            new_column = column + offset
            if 0 <= new_column < self.board_size:
                adjacent.append((row, new_column))

            new_row = row + offset
            if 0 <= new_row < self.board_size:
                adjacent.append((new_row, column))

        # Return a list of tuples containing the row and column of each adjacent space for a given space        
        return adjacent
    
    # Get the total combinations of stone trails possible for each move action
        # Example, moving a stack of 3 stones result in the following possible trails
        # (2, 1)
        # (1, 2)
        # This excludes the case of just moving all 3 stones into the immediate adjacent space which is accounted for in the takActions method

    # Method created with the help of ChatGPT 3.5
    @time_tracker
    def generate_combinations(self, stones_carried, result):

        if stones_carried == 0:
            return result
        
        # Initialize the max tuple size depending on the board size
        if stones_carried == self.board_size:
            tuple_size = stones_carried - 1
        else:
            tuple_size = stones_carried

        def generate_combinations_helper(current_combination, remaining_sum, remaining_elements):
            if remaining_elements == 0:
                if remaining_sum == 0:
                    result.append(tuple(current_combination))
                return

            for i in range(remaining_sum + 1):
                new_combination = current_combination + [i]
                generate_combinations_helper(new_combination, remaining_sum - i, remaining_elements - 1)

        
        generate_combinations_helper([], stones_carried, tuple_size)

        self.generate_combinations(stones_carried - 1, result)
    
        result = [tup for tup in result if not any(elem == 0 and i < len(tup) - 1 and tup[i + 1] > 0 for i, elem in enumerate(tup))]
        
        # Return a list of tuples containing the possible combinations of trails given the move action
        return result
    
    # Map the discrete action integer to an action in tuple form
    @time_tracker
    def map_to_action(self, action):
        return self.actions[action-1]

    # Given a place action in tuple form, place a stone on the board
    @time_tracker
    def place_stone(self, action):
        
        # Unpack action
        _, space, type_, _ = action

        # Row and column to place stone
        row, column = space

        # If player 2, turn type into even
        if self.state['Player: '] == -1:
            type_ = type_ + 1

        # Place stone into space
        self.state['Board: '][row][column] = [type_]

    # Given a move action in tuple form, plae a stone on the board 
    @time_tracker
    def move_stone(self, action):

        # Unpack Action
        _, move_from, move_to, trail, _ = action

        # Determine the length of the move from the given trail
        num_stones_to_move = sum([x for x in trail])
        spaces_to_move = sum(1 for stack in trail if stack != 0)
        
        # Determine the stones that are being moved from the inital space
        stones_to_move = [self.state['Board: '][move_from[0]][move_from[1]][i] for i in range(0, num_stones_to_move)]
        
        # Get the direction of the move
        direction = self.determine_direction(move_from, move_to)

        # If changing columns, moving left or right
        if (direction == "right") or (direction == "left"):
        
            # Move stack, leaving pieces in spaces corresponding to the trail
            for space in range(1, spaces_to_move + 1):
                
                stones_dropped = trail[space - 1]

                if direction == 'right':
                    previous_stack = self.state['Board: '][move_from[0]][move_from[1] + space]

                    self.state['Board: '][move_from[0]][move_from[1] + space] = stones_to_move[0:stones_dropped] + previous_stack

                else:
                    previous_stack = self.state['Board: '][move_from[0]][move_from[1] - space]

                    self.state['Board: '][move_from[0]][move_from[1] - space] = stones_to_move[0:stones_dropped] + previous_stack
                
                # Remove stones that have been dropped from the stone left to be moved
                for _ in range(0, stones_dropped):
                    stones_to_move.pop(0)

        # Changing rows, moving up or down
        else:
        
        # Move stack, leaving pieces corresponding to the trail
            for space in range(1, spaces_to_move + 1):
                
                stones_dropped = trail[space - 1]

                if direction == 'up':
                    previous_stack = self.state['Board: '][move_from[0] + space][move_from[1]] 
                
                    self.state['Board: '][move_from[0] + space][move_from[1]] = stones_to_move[0:stones_dropped] + previous_stack

                else:
                    previous_stack = self.state['Board: '][move_from[0] - space][move_from[1]] 
                
                    self.state['Board: '][move_from[0] - space][move_from[1]] = stones_to_move[0:stones_dropped] + previous_stack
                
                # Remove stones that have been dropped from the stone left to be moved
                for _ in range(0,stones_dropped):
                    stones_to_move.pop(0)
        
        # Remove stones that have been moved from the initial space
        for _ in range(0, num_stones_to_move):
            self.state['Board: '][move_from[0]][move_from[1]].pop(0)

    # Determine the direction when moving between two spaces, assuming the move is only horizontal or vertical, not diagonal
    @time_tracker
    def determine_direction(self, initial_position, final_position):

        # Extract row and column indices from the tuples
        initial_row, initial_column = initial_position
        final_row, final_column = final_position

        # If there is no change in the row index, column is changing, moving left or right
        if initial_row == final_row:

            if initial_column < final_column:
                return "right"
            else:
                return "left"
        
        # Else row is changing, moving up or down
        else:
            if initial_row < final_row:
                return "up"
            else:
                return "down"

    # Create a list of valid actions from all possible actions
    @time_tracker
    def get_valid_actions(self):
        valid_actions = [action for action in self.actions if self.valid_action(action)]
        return valid_actions

    # Determine of the action is valid
    @time_tracker
    def valid_action(self, action):

        if not isinstance(action, tuple):
            return False

        # If the 'Place' action
        if action[0] == 2:
            # Unpack action
            _, move_from, _, _ = action
        
            row, column = move_from
            
            # If the space is empty, return True
            if self.state['Board: '][row][column] == []:
                
                return True
            
            # If space is occupied, return False
            else:
                return False
            
        # If 'Move' action
        else:
            # Unpack action
            _, move_from, move_to, trail, _ = action
        
            row, column = move_from
            
            # If space is empty, return false
            if self.state['Board: '][row][column] == []:
                return False
            
            # Determine if the player controls the stack
            player_control = True
            
            # If the top stone is an even number, then it is controlled be player 2
            if self.state['Board: '][row][column][0] % 2 == 0:
                if self.state['Player: '] == 1:
                    player_control = False
                    
            # Else the top stone is controlled by player 1
            else:
                if self.state['Player: '] == -1:
                    player_control = False
                
            if player_control == False:
                return False
            
            # Determine if a standing stone if capstone is in the path of the move
            standing_stone_in_path = False
            
            # Check the spaces from the initial space to the final space
            for space in self.get_spaces_moved_in(move_from, move_to):
                row_check, column_check = space 
                
                # If the space contains a stone
                if self.state['Board: '][row_check][column_check] != []:

                    # If the space contains a standing stone of capstone
                    if (self.state['Board: '][row_check][column_check][0] == 3) or (self.state['Board: '][row_check][column_check][0] == 4)\
                          or (self.state['Board: '][row_check][column_check][0] == 5) or (self.state['Board: '][row_check][column_check][0] == 6):
                        standing_stone_in_path = True
            
            if standing_stone_in_path == True:
                return False
            
            # Determine if the correct number of stones are being moved
            moved_stones = 0

            # Get total number of stones being moved
            for stones in trail:
                moved_stones += stones

            if len(self.state['Board: '][row][column]) < moved_stones:
                return False
            
        # If all checks are passed, return True
        return True

    # Determine the spaces that are being moved, assuming the move is in either the horizontal or vertical direction
    @time_tracker
    def get_spaces_moved_in(self, initial_position, final_position):
        initial_row, initial_column = initial_position
        final_row, final_column = final_position
        
        # If row changing
        if initial_row == final_row:
            spaces = abs(final_column - initial_column)
            
            if initial_column < final_column:
                return [(initial_row, initial_column + i) for i in range(1, spaces + 1)]
            else:
                return [(initial_row, initial_column - i) for i in range(1, spaces + 1)]
            
        # Else column is changing
        else:
            spaces = abs(final_row - initial_row)

            if initial_row < final_row:
                return [(initial_row + i, initial_column) for i in range(1, spaces + 1)]
            else:
                return [(initial_row - i, initial_column) for i in range(1, spaces + 1)]
            
    # Determine if all spaces are filled
    @time_tracker
    def board_filled(self):
        return all(all(space) for space in self.state['Board: '])

    # Determine is a road has been made
    # Method created with the help of ChatGPT 3.5
    @time_tracker
    def is_road(self, player):

        # Determine which player's stones are going to be evaluated
        if player == 1:
            flat_stones_to_check = 1
            capstone_to_check = 5
        else:
            flat_stones_to_check = 2
            capstone_to_check = 6

        # Create a board represented by the top of each stack in each space
        board = self.update_top_stones()

        # Function to perform depth-first search
        def dfs(x, y, x_initial, y_initial):

            # Determine if a road connects the left and right sides of the board 
            if x_initial == 0:
                if x == board.shape[0]-1:
                    return True
            elif x_initial == board.shape[0]-1:
                if x == 0:
                    return True
                
            # Determine if a road connects the top and bottom of the board
            if y_initial == 0:
                if y == board.shape[0]-1:
                    return True
            elif y_initial == board.shape[0]-1:
                if y == 0:
                    return True

            # Define possible moves (in all 4 directions: up, down, left, right)
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # Check the adjacent sides of each space
            for dx, dy in moves:
                new_x, new_y = x + dx, y + dy

                if self.is_valid_move(board, new_x, new_y) and (board[new_x, new_y] == flat_stones_to_check):
                    
                    # Mark visited by changing the player's stone temporarily
                    board[x, y] = -1

                    # Recursively explore the next position
                    if dfs(new_x, new_y, x_initial, y_initial):
                        return True
                    
            return False
        
        # Check spaces where the top of each stack is a flat stone or capstone controlled by the player
        flat_stones = np.argwhere(board == flat_stones_to_check)
        
        for flat_stone in flat_stones:
    
            x, y = flat_stone
    
            if dfs(x, y, x, y):
                return True
    
        return False

    # Helper for is_road method, Checks if the move is within the board boundaries
    @time_tracker
    def is_valid_move(self, board, x, y):
        return 0 <= x < board.shape[0] and 0 <= y < board.shape[1]
 
    # Retrieve the top stone of each stack from each space
    @time_tracker
    def update_top_stones(self):

        # Initialize an empty board of the top stones
        top_stones = np.empty((self.board_size, self.board_size), dtype=list)
        top_stones.fill([])

        # Get the top stone of each stack in each space
        for row in range(0, self.board_size):
            for column in range(0, self.board_size):
                if self.state['Board: '][row][column] == []:
                    top_stones[row][column] = 0
                else:
                    top_stones[row][column] = self.state['Board: '][row][column][0]

        return top_stones
            