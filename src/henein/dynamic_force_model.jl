using DrWatson
using Agents
using Random
using Distributions
using AgentsPlots
using StaticArrays
using LinearAlgebra

mutable struct agent <: AbstractAgent
    id::Int64
    pos::Tuple{Int64,Int64}
    movetopos::Tuple{Int64,Int64}
    pushforce::Int64
    isinjured::Bool
end

agent(id, pos; movetopos, pushforce, isinjured) = agent(id, pos, movetopos, pushforce, isinjured)

mutable struct cell <: AbstractAgent
    id::Int64
    pos::Tuple{Int64,Int64}
    d_bosons::Int64
    isdoorentry::Bool
    isoccupied::Bool
    field_strength::Float64
    push_direction::Vector{Float64}
    f_bosons::Int64
    f_bosons_p::Int64
end

function make_static_field(grid_dims = (30, 30), door_pos = [12, 0])
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

function crowd_abm_henein_df(;
    density = 0.3,
    grid_dims = (30, 30),
    delta = 0.5,
    alpha = 0.3,
    kD = 0.09,
    kS = 1.0,
    door_pos = [12, 0],
    pf_m = 5.0,
    pf_sd = 1.0,
    forced_selection = 3,
    inj_thresh = 30,
    occupy_coef = 0.5,
)
    space = GridSpace(grid_dims; moore = false)
    door_entry = get_door_entry(grid_dims, door_pos)
    s_field = make_static_field(grid_dims, door_pos)
    properties = Dict(
        :delta => delta,
        :alpha => alpha,
        :kD => kD,
        :kS => kS,
        :d_ps => door_entry,
        :f_thres => forced_selection,
        :inj_thresh => inj_thresh,
        :dims => grid_dims,
        :occupy_coef => occupy_coef,
    )
    model = AgentBasedModel(Union{agent,cell}, space; properties = properties)
    _idx = 1
    push_dist = truncated(Normal(pf_m, pf_sd), 0, Inf)

    for x = 1:grid_dims[1]
        for y = 1:grid_dims[2]
            if [x, y] == door_entry
                Cell = cell(_idx, (x, y), 0, true, false, s_field[x, y], [0.0, 0.0], 0, 0)
                add_agent_pos!(Cell, model)
            else
                Cell = cell(_idx, (x, y), 0, false, false, s_field[x, y], [0.0, 0.0], 0, 0)
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
            Agent = agent(_idx, Cell.pos, Cell.pos, ceil(rand(push_dist)), false)
            add_agent_pos!(Agent, model)
            _idx += 1
        end
    end
    return model
end

function d_boson_step(agent::cell, model)
    n_bos = agent.d_bosons
    neighbours = get_neighbouring_cells(agent, model)
    t_n_bos = n_bos
    for i = 1:t_n_bos
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

function get_current_cell(agent, model)
    cc = filter(x -> isa(x, cell), get_node_agents(agent.pos, model))[1]
    return cc
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

function move_due_to_force(agent, Cell, model)
    direction = Cell.push_direction
    @assert Cell.f_bosons >= model.f_thres
    position = agent.pos

    right_weight = abs2(direction[1])
    up_weight = abs2(direction[2])

    go_lateral = right_weight >= up_weight

    if go_lateral
        #resolve laterally
        if direction[1] > 0
            if position[1] < model.dims[1] && position[1] > 1
                to_node = filter(x -> isa(x, cell), get_node_agents((position[1] + 1, position[2]), model))[1]
                return to_node.pos
            end
        else
            if position[1] < model.dims[1] && position[1] > 1
                to_node = filter(x -> isa(x, cell), get_node_agents((position[1] - 1, position[2]), model))[1]
                return to_node.pos
            end
        end
    else
        #resolve longitudinally
        if direction[2] > 0
            if position[2] < model.dims[2] && position[2] > 1
                to_node = filter(x -> isa(x, cell), get_node_agents((position[1], position[2] + 1), model))[1]
                return to_node.pos
            end
        else
            if position[2] < model.dims[2] && position[2] > 1
                to_node = filter(x -> isa(x, cell), get_node_agents((position[1], position[2] - 1), model))[1]
                return to_node.pos
            end
        end
    end
end

function direct_force_bosons(agent::cell, model)
    n_force_bosons = agent.f_bosons
    if n_force_bosons > 0
        direction = agent.push_direction
        position = agent.pos

        if !isapprox(direction, [0.0, 0.0], atol = 1e-4)
            right_weight = abs2(direction[1])
            up_weight = abs2(direction[2])
            # println(agent)
            s_weight = right_weight + up_weight
            right_weight /= s_weight
            up_weight /= s_weight

            #1 denotes right, 2 denotes up
            p_outcomes = [1, 2]
            #println([right_weight, up_weight])
            p_dist = DiscreteNonParametric(p_outcomes, [right_weight, up_weight])
            outcomes = rand(p_dist, n_force_bosons)
            tot_lat = count(x -> x == 1, outcomes)
            tot_lon = count(x -> x == 2, outcomes)
            #resolve laterally
            if direction[1] > 0
                if position[1] < model.dims[1] && position[1] > 1
                    to_node = filter(x -> isa(x, cell), get_node_agents((position[1] + 1, position[2]), model))[1]
                    to_node.f_bosons += tot_lat
                    agent.f_bosons_p -= tot_lat
                end
            else
                if position[1] < model.dims[1] && position[1] > 1
                    to_node = filter(x -> isa(x, cell), get_node_agents((position[1] - 1, position[2]), model))[1]
                    to_node.f_bosons += tot_lat
                    agent.f_bosons -= tot_lat
                end
            end

            #resolve longitudinally
            if direction[2] > 0
                if position[2] < model.dims[2] && position[2] > 1
                    to_node = filter(x -> isa(x, cell), get_node_agents((position[1], position[2] + 1), model))[1]
                    to_node.f_bosons += tot_lon
                    agent.f_bosons_p -= tot_lon
                end
            else
                if position[2] < model.dims[2] && position[2] > 1
                    to_node = filter(x -> isa(x, cell), get_node_agents((position[1], position[2] - 1), model))[1]
                    to_node.f_bosons += tot_lon
                    agent.f_bosons_p -= tot_lon
                end
            end
        else
        end
    end
end

function determine_next_cell(agent::agent, model)
    current_cell = get_current_cell(agent, model)
    if current_cell.f_bosons > model.f_thres * agent.pushforce
        agent.movetopos = move_due_to_force(agent, current_cell, model)
    else
        neighbour_ids = get_neighbouring_cells(agent, model)
        neighbour_cells = [model[x] for x in neighbour_ids]
        occupy_coef_vec = [x.isoccupied ? 1.0 : model.occupy_coef for x in neighbour_cells]
        n_valid = length(neighbour_cells)
        p_ij = Vector{Float64}(undef, n_valid)
        n_bosons = Vector{Int64}(undef, n_valid)
        for n = 1:n_valid
            p_ij[n] = neighbour_cells[n].field_strength
            n_bosons[n] = neighbour_cells[n].d_bosons
        end
        p_ij = exp.(p_ij .* model.kS) .* exp.(n_bosons .* model.kD) .* occupy_coef_vec
        p_ij ./= sum(p_ij)
        #println(p_ij)
        dist = DiscreteNonParametric(collect(1:(n_valid)), p_ij)
        atm = rand(dist)
        attempt_move_to = neighbour_cells[atm].pos
        agent.movetopos = attempt_move_to
    end
end

function propagate_force(agent::cell, model)
    if !agent.isoccupied
        agent.f_bosons = 0
    else
        agent.f_bosons = agent.f_bosons_p
        agent.f_bosons_p = 0
    end
end

function crowd_agent_move!(agent::agent, model)
    desired_cell = agent.movetopos
    dc_contents = get_node_agents(desired_cell, model)
    if length(dc_contents) == 1
        get_current_cell(agent, model).d_bosons += 1
        get_current_cell(agent, model).isoccupied = false
        move_agent!(agent, desired_cell, model)
        get_current_cell(agent, model).isoccupied = true
    else
        push_cell = dc_contents[1]
        push_vector = [push_cell.pos[1] - agent.pos[1], push_cell.pos[2] - agent.pos[2]]
        pp_vector = push_cell.push_direction * push_cell.f_bosons
        pp_vector_new = pp_vector + agent.pushforce * push_vector
        if norm(pp_vector_new) > 0
            pp_vector_new ./= norm(pp_vector_new)
        end
        push_cell.push_direction = pp_vector_new
        push_cell.f_bosons += agent.pushforce
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
    for cell in cells
        direct_force_bosons(cell, model)
    end
    for cell in cells
        propagate_force(cell, model)
    end
end

mcolor(a::cell) = RGBA(0.0, 1.0, 0.0, 0.0)
mcolor(a::agent) = RGBA(1.0, 0.0, 0.0, 0.0)
# asize(a::cell) = a.isdoorentry ? 5.0 : 0.0
asize(a::cell) = a.f_bosons
# asize(a::cell) = a.d_bosons * 1.
asize(a::agent) = 5.0

t_model = crowd_abm_henein_df(; density = 0.45, grid_dims = (40, 40), door_pos = [20, 0], kD = 0.7, kS = 1.0)
plotabm(t_model; ac = mcolor, as = asize)

step = 10
frames = 50

gr()

anim = @animate for i = 0:step:(step*frames)
    i > 0 && step!(t_model, dummystep, crowd_model_step!, step)
    p1 = plotabm(t_model; ac = mcolor, as = asize)
    title!(p1, "step $(i)")
end

gif(anim, "ne_he_ts_crowd.gif", fps = 10)
