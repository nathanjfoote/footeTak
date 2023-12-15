using Printf
using PyCall
include("footeMCTS.jl")
gym = pyimport("gym")
foote_tak = pyimport("foote_tak")

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
    
    env_play = gym.make("Tak-v0");
    env_render = gym.make("Tak-v0", render_mode="human")
    env_play.reset();
    env_render.reset();
    
    
    solver = MCTS_Solver(50, 1.25, 19_652);
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
                _, act = footeMCTS.solve(solver, env_play)
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