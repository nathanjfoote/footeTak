using POMDPs
using PyCall
using Printf
using StatsBase
using ProgressMeter

gym = pyimport("gym")
foote_tak = pyimport("foote_tak")

struct MCTS_Solver <: Solver
    simulations::Int
    exploration_constant_init::Float64
    exploration_constant_base::Float64
end

function MCTS_Solver(; simulations=100)
    return MCTS_Solver(simulations)
end

mutable struct MCTSNode
    state::PyObject
    parent::Union{MCTSNode, Nothing}
    children::Vector{MCTSNode}
    visit_count::Int
    total_value::Float64
    mean_value::Float64
    prior::Float64
    action
    level::Int
end

function simulate!(
        node::MCTSNode, 
        env, 
        exploration_constant_init::Float64, 
        exploration_constant_base::Float64, 
        depth=20
    )

    total_reward = 0.0
    action_space = env.get_valid_actions()

    if env.is_terminal()
        
        return 0.0
    elseif depth == 0
        return rollout(node, env)
    end

    # Check if the node is already fully expanded
    if !is_fully_expanded(node, action_space)
        # Iterate over possible actions
        for action in action_space

            # Check if this action is already represented in the children of the node
            if !has_child_for_action(node, action)

                # Generate the resulting state from taking this action
                new_state = generate_state(action, env)

                # Create a new child node
                new_child = MCTSNode(
                    new_state,                  # The state from the environment
                    node,                       # Record previous node as the parent
                    Vector{MCTSNode}(),         # Initially no childern
                    0,                          # No Visits
                    0.0,                        # Initial Value
                    0.0,                        # mean_value
                    1.0 / length(action_space), # probability
                    action,                     # Action that leads to the node
                    node.level + 1              # Depth of tree
                )

                # Add this child to the node's children
                push!(node.children, new_child)

            end
        end
        
        return rollout(node, env)
    end

    best_child = select_best_child(node, exploration_constant_init, exploration_constant_base)

    _, reward_collected, _, _ = env.step(best_child.action[end])
            
    total_reward = reward_collected + simulate!(best_child, env, exploration_constant_init, exploration_constant_base, depth - 1)

    # Update the visit count
    node.visit_count += 1

    # Update the total value with the reward from the simulation
    node.total_value += total_reward

    node.mean_value = node.total_value / node.visit_count

    return total_reward
end

function POMDPs.solve(solver::MCTS_Solver, env)

    # Initialize the root node of the tree
    root = MCTSNode(
        env.state,          # The state from the environment
        nothing,            #  No parent, as this is the root
        Vector{MCTSNode}(), # Initially, no children
        0,                  # No visits yet
        0.0,                # Initial value (could be zero or based on a heuristic)
        0.0,                # Mean value
        0.0,                # Prior value
        nothing,
        1
    )
    
    for _ in 1:solver.simulations
        scratch_env = env.clone()
        simulate!(root, scratch_env, solver.exploration_constant_init, solver.exploration_constant_base)
    end

    best_child = root.children[1]
    
    for child in root.children
        if child.visit_count > best_child.visit_count
            best_child = child
        end
    end
    return best_child.action
end

function rollout(node, env)
    terminated = false
    env_copy = env.clone()
    total_reward = 0
    
    while !terminated
        
        action = epsilon_greedy_action(node, env_copy, .9)[end]

        # Perform the action
        _, rollout_reward, terminated, _ = env_copy.step(action)

        # Accumulate the reward
        total_reward += rollout_reward

        # Break if a terminal state is reached
        if terminated
            break
        end
    end
    return total_reward
end

function select_best_child(node::MCTSNode, exploration_constant_init::Float64, exploration_constant_base::Float64)
    best_child = 0.0
    best_value = -Inf

    for child in node.children
        # Calculate the UCB value for the child
        ucb_value = uct_value(child, exploration_constant_init, exploration_constant_base)

        if ucb_value > best_value
            best_child = child
            best_value = ucb_value
        end
    end

    return best_child
end

function best_action(node::MCTSNode, env)
    # Check if the root has children
    if isempty(node.children)
        error("No actions to choose from: root node has no children.")
    end
    
    valid_children = filter(child -> env.valid_action(child.action), node.children)
    
    # Handle case where no valid actions are found
    if isempty(valid_children)
        return rand(env.get_valid_actions())
    end

    best_child_index = argmax(map(child -> child.visit_count, valid_children))

    return valid_children[best_child_index].action
end

function epsilon_greedy_action(node::MCTSNode, env, epsilon)
    
    available_actions = env.get_valid_actions()
    
    if rand() < epsilon
        # Exploration: choose a random action
        return rand(available_actions)
    else
        # Exploitation: choose the best-known action
        return best_action(node, env)
    end
end

function is_fully_expanded(node::MCTSNode, action_space)
    # If the number of children is equal to the number of actions, the node is fully expanded
    return length(node.children) == length(action_space)
end

function has_child_for_action(node::MCTSNode, action)
    # Iterate through the children of the node
    for child in node.children
        # Check if the child node's action matches the given action
        if child.action == action
            return true
        end
    end
    return false
end

function generate_state( action, env)
    
    # Make a copy of the environment
    env_copy = env.clone()
    
    # Perform the action on the copied environment
    env_copy.step(action[end])

    # Return the new state
    return env_copy.state
end

function uct_value(node::MCTSNode, exploration_constant_init::Float64, exploration_constant_base::Float64)        
    if node.visit_count == 0
        return Inf  # Assign infinite value to unvisited nodes to ensure they get selected
    else
        pb_c = log((1 + node.parent.visit_count + exploration_constant_base) / exploration_constant_base) + exploration_constant_init
        pb_c *= sqrt(node.parent.visit_count) / (node.visit_count + 1)
        
        prior_score = pb_c * node.prior
        value_score = node.mean_value
    end
        
        return prior_score + value_score
end

function printaction(action)
    
    if action[1] == 2   
        
        _, space, type_, _ = action
        
        if (type_ == 1) || (type_ == 2)
            type_ = "Flat Stone"    
        else
            type_ = "Standing Stone"
        end
        
        @printf "Place a %s into: (%i, %i)\n" type_ space[1] space[2]
    
    else
        _, move_from, move_to, trail, _ = action
        
        count = 0
        
        for i in trail
            count += i
        end
            
        piece = " piece"
        
        if count > 1
            piece = " pieces"
        end
        
        @printf "Move %i %s from: (%i, %i) to: (%i, %i)\n" count piece move_from[1] move_from[2] move_to[1] move_to[2]
    end
end

function random_play(env)
    action_space = env.get_valid_actions()
    return rand(action_space)
end

function play_tak(players::Dict{Int64, Symbol})
    
    env_play = gym.make("Tak-v0")
    env_render = gym.make("Tak-v0", render_mode="human")
    env_play.reset()
    env_render.reset()
    
    solver = MCTS_Solver(100, 1.25, 19_652)
    turn = 0
    terminated = false
    reward_collected = 0
    moves = []
    board = []
    
    while !terminated
        turn += 1

        player = env_play.state["Player: "]

        if  player == 1
            player_print = "Player 1"
        else
            player_print = "Player 2"
        end

        play_style = get(players, player, "")

        if  play_style == :random
                act = random_play(env_play)
        else
                act = solve(solver, env_play)
        end

        push!(moves, act)

        _, reward_collected, terminated, _ = env_play.step(act[end]);
        env_render.step(act[end]);
        push!(board, env_play.state)
    end
    
    if env_play.state["Player: "] == 1
        if reward_collected == 1
            winner = "Player 1"
        elseif reward_collected == -1
            winner = "Player 2"
        else
            winner = "Draw"
        end
    else
        if reward_collected == 1
            winner = "Player 2"
        elseif reward_collected == -1
            winner = "Player 1"
        else
            winner = "Draw"
        end
    end
        
    
    return moves, board, winner
end


function play_tak_no_render(players::Dict{Int64, Symbol})
    
    env_play = gym.make("Tak-v0")
    env_play.reset()
    
    solver = MCTS_Solver(100, 1.25, 19_652)
    turn = 0
    terminated = false
    reward_collected = 0
    moves = []
    board = []
    
    while !terminated
        turn += 1

        player = env_play.state["Player: "]

        if  player == 1
            player_print = "Player 1"
        else
            player_print = "Player 2"
        end

        play_style = get(players, player, "")

        if  play_style == :random
                act = random_play(env_play)
        else
                act = solve(solver, env_play)
        end

        push!(moves, act)

        _, reward_collected, terminated, _ = env_play.step(act[end]);

        push!(board, env_play.state)
    end
    
    if env_play.state["Player: "] == 1
        if reward_collected == 1
            winner = "Player 1"
        elseif reward_collected == -1
            winner = "Player 2"
        else
            winner = "Draw"
        end
    else
        if reward_collected == 1
            winner = "Player 2"
        elseif reward_collected == -1
            winner = "Player 1"
        else
            winner = "Draw"
        end
    end
        
    
    return moves, board, winner
end
