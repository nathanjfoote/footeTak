using PyCall
using Flux

# Define a custom layer for the skip connection
struct SkipConnection
    layer::Chain
end

Flux.@functor SkipConnection

function alphatak_input(env)
    # Extract player and board information
    player = [env.state["Player: "]]
    board_py = env.state["Board: "]
    
    # Define the input shapes
    player_input_shape = (1,)  # Single value for the player's turn
    board_input_shape = (env.board_size, env.board_size, env.pieces)  # 3x3 matrix with lists of up to 10 integers each

    # Flatten the board input shape for the neural network
    board_input_shape_flat = prod(board_input_shape)

    # Combine the input shapes
    total_input_shape = player_input_shape[1] + board_input_shape_flat

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
    total_input_length = total_input_shape

    total_input = length(total_input) > total_input_length ? total_input[1:total_input_length] : vcat(total_input, fill(0::Int64, total_input_length - length(total_input)))
    return total_input
end

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

function alphatak_nn()

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
        Conv((1,), channels => 128, relu),
        flattenLayer,
        Dense(128, 126),
        softmax
    )

    # Value Head
    valueHead = Chain(
        Conv((1,), channels => 128, relu),
        flattenLayer,
        Dense(128, 1),
        tanh
    )

    # Combine heads with the shared body
    return Chain(inputLayer, reshapeLayer, resNetLayers, x -> (policyHead(x), valueHead(x)))
end