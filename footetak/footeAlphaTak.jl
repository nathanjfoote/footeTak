using POMDPs
using PyCall
using Printf
using StatsBase
using ProgressMeter
using Flux

gym = pyimport("gym")
foote_tak = pyimport("foote_tak")

@kwdef struct AlphaZeroConfig
    num_actors::Int64 = 5000
    num_simulations::Int64 = 800
    root_dirichlet_exploration_alpha::Float64 = 0.3
    root_exploration_fraction::Float64 = 0.25
    training_steps::Int64 = 1_000
    checkpoint_interval::Int64 = 1_000
    window_size::Int64 = 1_000_000
    batch_size::Int64 = 4096
    weight_decay::Int64 = 10_000
    learning_rate_schedule::Dict{Int64, Float64} = Dict{Int64, Float64}(
        0 => 2e-1,
        100e3 => 2e-2,
        300e3 => 2e-3,
        500e3 => 2e-4
    )
end

# Define a custom layer for the skip connection
struct SkipConnection
    layer::Chain
end

Flux.@functor SkipConnection

function (sc::SkipConnection)(x)
    return sc.layer(x) + x
end

# Define a basic block of the ResNet
function resBlock(channels::Int)
    convLayer1 = Conv((3,), channels => channels, pad = (1,), stride = 1)
    batchNormLayer1 = BatchNorm(channels, relu)
    convLayer2 = Conv((3,), channels => channels, pad = (1,), stride = 1)
    batchNormLayer2 = BatchNorm(channels, relu)
    
    block = Chain(convLayer1, batchNormLayer1, convLayer2, batchNormLayer2)
    return SkipConnection(block)
end

function create_network()
    # Number of channels
    channels = 64

    # Input layer
    inputLayer = Dense(91, channels)

    # Reshape layer
    reshapeLayer = x -> reshape(x, 1, length(x), 1)

    # Define the ResNet structure
    resNetLayers = Chain(
        [resBlock(channels) for _ in 1:3]... # Stacking 3 residual blocks
    )

    # Custom Flatten layer
    flattenLayer(x) = reshape(x, :, size(x, 4))


    # Policy Head
    policyHead = Chain(
        Conv((1,), channels => 64, relu),
        flattenLayer,
        Dense(64, 126),
        softmax
    )

    # Value Head
    valueHead = Chain(
        Conv((1,), channels => 64, relu),
        flattenLayer,
        Dense(64, 1),
        tanh
    )
    return Chain(inputLayer, reshapeLayer, resNetLayers, x -> (policyHead(x), valueHead(x)))
end

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
        env::PyObject, 
        exploration_constant_init::Float64, 
        exploration_constant_base::Float64,
        network, 
        depth::Int64=20
    )

    total_reward = 0.0
    action_space = env.get_valid_actions()

    if env.is_terminal()
        return 0.0
    elseif depth == 0
        return action_selection(env, network)[end]
    end

    # Check if the node is already fully expanded
    if !is_fully_expanded(node::MCTSNode, action_space)
        # Iterate over possible actions
        for action in action_space

            # Check if this action is already represented in the children of the node
            if !has_child_for_action(node::MCTSNode, action)

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
        
        return action_selection(env, network)[end]
    end

    best_child = select_best_child(node, exploration_constant_init, exploration_constant_base)

    _, reward_collected, _, _ = env.step(best_child.action[end])
            
    total_reward = reward_collected + simulate!(best_child, env, exploration_constant_init, exploration_constant_base, network, depth - 1)

    # Update the visit count
    node.visit_count += 1

    # Update the total value with the reward from the simulation
    node.total_value += total_reward

    node.mean_value = node.total_value / node.visit_count

    return total_reward
end

function POMDPs.solve(solver::MCTS_Solver, env, network)

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
        simulate!(root, scratch_env, solver.exploration_constant_init, solver.exploration_constant_base, network)
    end

    return root

    # best_child = root.children[1]
    
    # for child in root.children
    #     if child.visit_count > best_child.visit_count
    #         best_child = child
    #     end
    # end
    # return best_child.action
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

function alphatak_input(env)
    # Extract player and board information
    player = [env.state["Player: "]::Int]
    board_py = env.state["Board: "]::Array{PyObject, 2}
    
    # Define the input shapes
    player_input_shape = (1,)  # Single value for the player's turn
    board_input_shape = (env.board_size::Int, env.board_size::Int, env.pieces::Int)  # 3x3 matrix with lists of up to 10 integers each

    # Flatten the board input shape for the neural network
    board_input_shape_flat = prod(board_input_shape)::Int

    # Combine the input shapes
    total_input_shape = player_input_shape[1] + board_input_shape_flat::Int

    # Convert PyObject lists in the board to Julia arrays and flatten them
    board_flat = Int[]
    for cell in board_py
        cell_list = convert(Array{Int}, PyVector(cell))
        
        while length(cell_list) < 10
            push!(cell_list, 0)
        end
        
        append!(board_flat, cell_list)  # Flatten each cell's list into board_flat
    end

    # Combine player and board inputs
    total_input = vcat(player, board_flat)
    
    # Truncate or pad the input to match the expected input shape
    total_input_length = total_input_shape::Int

    total_input = length(total_input) > total_input_length ? total_input[1:total_input_length] : vcat(total_input, fill(0::Int64, total_input_length - length(total_input)))
    return total_input::Vector{Int}
end

function action_selection(env, network)
    input = alphatak_input(env)
    action_probabilities::Array{Float32, 2}, value::Matrix{Float32} = network(input)
    valid_action_indices = [act[end] for act in env.get_valid_actions()]::Vector{Int64}
    
    # Create a new array of zeros with the same length as move_probs
    adjusted_probabilities = zeros(length(action_probabilities))::Vector{Float64}

    # Set the probabilities of valid moves
    for idx in valid_action_indices
        adjusted_probabilities[idx] = action_probabilities[idx]
    end

    # Normalize the adjusted probabilities
    total_probability = sum(adjusted_probabilities)
    if total_probability > 0
        adjusted_probabilities ./= total_probability
    end
    
    return argmax(adjusted_probabilities), adjusted_probabilities, only(value)
end