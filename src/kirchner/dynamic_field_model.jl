using DrWatson
using Agents
using Random
using Distributions
using AgentsPlots

mutable struct agent <: AbstractAgent
    id::Int64
    pos::Tuple{Int64,Int64}
    movetopos::Tuple{Int64,Int64}
end

mutable struct cell <: AbstractAgent
    id::Int64
    pos::Tuple{Int64,Int64}
    d_bosons::Int64
    isdoorentry::Bool
    isoccupied::Bool
    field_strength::Float64
end

cell(id, pos; d_bosons, isdoorentry, isoccupied, field_strength) = cell(id, pos, d_bosons, isdoorentry, isoccupied, field_strength)

function make_static_field(grid_dims = (5, 5), door_pos = [2, 0])
    s_field = zeros(grid_dims)
    for col = 1:grid_dims[2], row = 1:grid_dims[1]
        s_field[row, col] = sqrt((col - door_pos[2])^2 + (row - door_pos[1])^2)
    end
    s_field = maximum(s_field) .- s_field
    return s_field
end

function get_door_entry(grid_dims, door_pos)
    #top edge
    if door_pos[2] == 0
        return [door_pos[1], 1]
        #bottom egde
    elseif door_pos[2] == grid_dims[2] + 1
        return [door_pos[1], grid_dims[2]]
        #left edge
    elseif door_pos[1] == 0
        return [1, door_pos[2]]
        #right edge
    elseif door_pos[1] == grid_dims[1] + 1
        return [grid_dims[1], door_pos[2]]
    end
end

function crowd_abm_kirchner_df(; density = 0.3, grid_dims = (30, 30), delta = 0.2, alpha = 0.2, moore = false, kD = 0.09, kS = 1.0, door_pos = [0, 15])
    space = GridSpace(grid_dims; moore = moore)
    door_entry = get_door_entry(grid_dims, door_pos)
    s_field = make_static_field(grid_dims, door_pos)
    properties = Dict(:delta => delta, :alpha => alpha, :kD => kD, :kS => kS, :d_ps => door_entry)
    model = AgentBasedModel(Union{agent,cell}, space; properties = properties)
    _idx = 1
    for x = 1:grid_dims[1]
        for y = 1:grid_dims[2]
            if [x, y] == door_entry
                Cell = cell(_idx, (x, y), 0, true, false, s_field[x, y])
                add_agent_pos!(Cell, model)
            else
               Cell = cell(_idx, (x, y), 0, false, false, s_field[x, y])
               add_agent_pos!(Cell, model)
            end
            _idx += 1
        end
    end
    #add person agents
    for node in nodes(model)
        if rand() <= density
            Cell = model[get_node_contents(node, model)[1]]
            Cell.isoccupied = true
            Agent = agent(_idx, Cell.pos, Cell.pos)
            add_agent_pos!(Agent, model)
            _idx += 1
        end
    end
    return model
end



function determine_next_cell(agent::cell, model)

end

function get_neighbouring_cells(agent, model)
    neighbours_c = node_neighbors(agent, model)
    cellids = Vector(undef, length(neighbours_c))
    cid = 1
    for neighbour in neighbours_c
        contents = minimum(get_node_contents(neighbour, model))
        cellids[cid] = contents
        cid += 1
    end
    return cellids
end

function get_current_cell(agent, model)
    cc = filter(x -> isa(x, cell), get_node_agents(agent.pos, model))[1]
    return cc
end

function determine_next_cell(agent::agent, model)
    neighbour_ids = get_neighbouring_cells(agent, model)
    neighbour_cells = [model[x] for x in neighbour_ids]
    valid_cells = filter(x -> !x.isoccupied, neighbour_cells)
    n_valid = length(valid_cells)
    p_ij = Vector{Float64}(undef, n_valid+1)
    n_bosons = Vector{Int64}(undef, n_valid+1)
    #sumtotal = 0.
    for n in 1:n_valid
        p_ij[n] = valid_cells[n].field_strength
        n_bosons[n] = valid_cells[n].d_bosons
        #sumtotal += p_ij[n]
    end
    current_cell = filter(x -> isa(x, cell), get_node_agents(agent.pos, model))[1]
    p_ij[n_valid+1] = current_cell.field_strength
    n_bosons[n_valid+1] = current_cell.d_bosons
    p_ij = exp.(p_ij .* model.kS) .* exp.(n_bosons .* model.kD)
    p_ij ./= sum(p_ij)
    # println(p_ij)
    # println(collect(1:(n_valid+1)))
    dist = DiscreteNonParametric(collect(1:(n_valid+1)), p_ij)
    atm = rand(dist)
    # print(atm)
    if atm <= n_valid
        attempt_move_to = valid_cells[atm].pos
        agent.movetopos = attempt_move_to
    else
        agent.movetopos = agent.pos
    end
end

function d_boson_step(agent::cell, model)
    n_bos = agent.d_bosons
    neighbours = get_neighbouring_cells(agent, model)
    t_n_bos = n_bos
    for i in 1:t_n_bos
        if rand() < model.delta
            n_bos -= 1
        elseif rand() < model.alpha
            n_bos -= 1
            movetoid = rand(neighbours, 1)[1]
            model[movetoid].d_bosons += 1
        end
    end
    agent.d_bosons = n_bos
end

function crowd_agent_move!(agent::agent, model)
    desired_cell = agent.movetopos
    dc_contents = get_node_agents(desired_cell, model)
    if length(dc_contents) == 1
        get_current_cell(agent, model).d_bosons += 1
        get_current_cell(agent, model).isoccupied = false
        move_agent!(agent, desired_cell, model)
        get_current_cell(agent, model).isoccupied = true
    end
end

function crowd_model_step!(model)
    allevery = collect(allagents(model))
    agents = filter(x -> isa(x, agent), allevery)
    cells = filter(x -> isa(x, cell), allevery)

    for cell in cells
        d_boson_step(cell, model)
    end
    for agent in agents
        determine_next_cell(agent, model)
    end
    for agent in shuffle(agents)
        crowd_agent_move!(agent, model)
        if agent.pos == (model.d_ps[1], model.d_ps[2])
            get_current_cell(agent, model).isoccupied = false
            kill_agent!(agent, model)
        end
    end
end

function terminate(model, s)
    allevery = collect(allagents(model))
    agents = filter(x -> isa(x, agent), allevery)
    if length(allagents) == 0
        return true
    else
        return false
    end
end

mcolor(a::cell) = RGBA(0., 1., 0., 0.)
mcolor(a::agent) = RGBA(1., 0., 0., 0.)
asize(a::cell) = a.isdoorentry ? 5. : 0.
# asize(a::cell) = a.d_bosons * 5
asize(a::agent) = 15.

t_model = crowd_abm_kirchner_df(; density = 0.3, grid_dims = (50, 10), door_pos = [25, 0], kD = 0.7, kS = 1.)
plotabm(t_model; ac = mcolor, as = asize)

step = 2
frames = 100

gr()

anim = @animate for i = 0:step:(step*frames)
    i > 0 && step!(t_model, dummystep, crowd_model_step!, step)
    p1 = plotabm(t_model; ac = mcolor, as = asize)
    title!(p1, "step $(i)")
end

gif(anim, "ts_crowd.gif", fps = 30)
