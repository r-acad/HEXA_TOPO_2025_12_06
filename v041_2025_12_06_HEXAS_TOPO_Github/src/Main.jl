# ==============================================================================
# 1. AUTO-INSTALL PACKAGES (GLOBAL / LATEST)
# ==============================================================================
using Pkg

# Define the root of the project (one level up from src)
const PROJECT_ROOT = joinpath(@__DIR__, "..")

const REQUIRED_PACKAGES = [
    "CUDA", 
    "JSON", 
    "JSON3", 
    "Krylov", 
    "LinearOperators", 
    "MarchingCubes", 
    "YAML"
]

function check_and_install_packages()
    println(">>> [SETUP] Checking global dependencies...")
    
    installed_set = Set(pkg.name for pkg in values(Pkg.dependencies()))
    
    for pkg_name in REQUIRED_PACKAGES
        if !(pkg_name in installed_set)
            println(">>> [SETUP] Package '$pkg_name' is missing. Installing latest version...")
            try
                Pkg.add(pkg_name)
            catch e
                println("!!! Error installing $pkg_name: $e")
            end
        end
    end
    println(">>> [SETUP] Dependencies ready.")
end

check_and_install_packages()


# ==============================================================================
# 2. MAIN APPLICATION
# ==============================================================================

println("\n>>> SCRIPT START: Loading Modules...")

module HEXA

using LinearAlgebra
using SparseArrays
using Printf
using Base.Threads
using JSON
using Dates
using Statistics 
using CUDA
using YAML

# Import from parent scope
using ..Main: PROJECT_ROOT

include("Utils/Diagnostics.jl")
include("Utils/Helpers.jl")
using .Diagnostics
using .Helpers

include("Core/Element.jl")
include("Core/Boundary.jl")
include("Core/Stress.jl")

using .Element
using .Boundary
using .Stress


include("Mesh/Mesh.jl")
include("Mesh/MeshUtilities.jl")
include("Mesh/MeshPruner.jl") 
include("Mesh/MeshRefiner.jl") 
include("Mesh/MeshShapeProcessing.jl") 

using .Mesh
using .MeshUtilities
using .MeshPruner 
using .MeshRefiner 
using .MeshShapeProcessing


include("Solvers/CPUSolver.jl")
include("Solvers/GPUSolver.jl")
include("Solvers/DirectSolver.jl")
include("Solvers/IterativeSolver.jl")
include("Solvers/Solver.jl") 

using .CPUSolver
using .GPUSolver
using .DirectSolver
using .IterativeSolver
using .Solver


include("IO/Configuration.jl")
include("IO/ExportVTK.jl")
include("IO/Postprocessing.jl")
include("Optimization/GPUHelmholtz.jl") 
include("Optimization/TopOpt.jl") 

using .Configuration
using .ExportVTK
using .Postprocessing
using .TopologyOptimization 

function __init__()
    Diagnostics.log_status("HEXA Finite Element Solver initialized")
    Helpers.clear_gpu_memory()
end

function run_main(config_file=nothing)
    try
        _run_safe(config_file)
    catch e
        println("\n" * "!"^60)
        println("!!! FATAL ERROR DETECTED !!!")
        println("!"^60)
        showerror(stderr, e, catch_backtrace())
    end
end

function _run_safe(config_file)
    if config_file === nothing
        config_file = joinpath(PROJECT_ROOT, "configs", "default.yaml")
    end
    
    println("Loading configuration from: $config_file")
    if !isfile(config_file)
        error("Configuration file not found: $config_file")
    end

    config = load_configuration(config_file)
    out_settings = get(config, "output_settings", Dict())
    export_freq = get(out_settings, "export_frequency", 5)
    
    # --- OUTPUT SETUP ---
    
    # 1. Define RESULTS directory for heavy files (VTK, STL, BIN)
    RESULTS_DIR = joinpath(PROJECT_ROOT, "RESULTS")
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    # 2. Define Log File Path (Root directory, next to Project.toml)
    # We ignore the folder in the config if it tries to put it elsewhere, 
    # enforcing the root location as requested.
    raw_log_name = get(out_settings, "log_filename", "simulation_log.txt")
    log_filename = joinpath(PROJECT_ROOT, basename(raw_log_name))
    
    iso_threshold_val = get(out_settings, "iso_surface_threshold", 0.8)
    iso_threshold = Float32(iso_threshold_val)
    
    Diagnostics.init_log_file(log_filename, config)

    # --- INITIAL MESH GENERATION ---
    geom = setup_geometry(config)
    nodes, elements, dims = generate_mesh(
        geom.nElem_x, geom.nElem_y, geom.nElem_z;
        dx = geom.dx, dy = geom.dy, dz = geom.dz
    )
    
    initial_target_count = size(elements, 1)
    
    domain_bounds = (
        min_pt = [0.0f0, 0.0f0, 0.0f0],
        len_x = geom.dx * geom.nElem_x,
        len_y = geom.dy * geom.nElem_y,
        len_z = geom.dz * geom.nElem_z
    )

    config["geometry"]["nElem_x_computed"] = geom.nElem_x
    config["geometry"]["nElem_y_computed"] = geom.nElem_y
    config["geometry"]["nElem_z_computed"] = geom.nElem_z
    config["geometry"]["dx_computed"] = geom.dx
    config["geometry"]["dy_computed"] = geom.dy
    config["geometry"]["dz_computed"] = geom.dz
    config["geometry"]["max_domain_dim"] = geom.max_domain_dim
    
    nNodes = size(nodes, 1)
    bc_data = config["boundary_conditions"]
    bc_indicator = get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
    
    E = Float32(config["material"]["E"])
    nu = Float32(config["material"]["nu"])
    
    ndof = nNodes * 3
    F = zeros(Float32, ndof)
    forces_data = config["external_forces"]
    apply_external_forces!(F, Vector{Any}(forces_data), nodes, elements)
    
    density, original_density, protected_elements_mask = 
        initialize_density_field(nodes, elements, geom.shapes_to_add, geom.shapes_to_remove, config)
    
    opt_params = config["optimization_parameters"]
    min_density = Float32(get(opt_params, "min_density", 1.0e-3))
    max_density_clamp = Float32(get(opt_params, "density_clamp_max", 1.0))

    base_name = splitext(basename(config_file))[1]
    
    # --- PHASE CONTROL SETTINGS ---
    nominal_iterations = get(config, "number_of_iterations", 30)
    
    # Sanitize input for target_active_elements
    raw_active_target = get(config, "target_active_elements", initial_target_count)
    final_target_active = if isa(raw_active_target, String)
        parse(Int, replace(raw_active_target, "_" => ""))
    else
        Int(raw_active_target)
    end

    growth_steps = get(config, "growth_phase_steps", 10)
    
    l1_stress_allowable = Float32(get(config, "l1_stress_allowable", 1.0))
    if l1_stress_allowable == 0.0f0; l1_stress_allowable = 1.0f0; end

    U_full = zeros(Float32, ndof)
    max_change = 1.0f0
    filter_R = 0.0f0
    curr_threshold = 0.0f0
    
    iter = 1
    keep_running = true
    is_annealing = false
    convergence_threshold = 0.005 
    
    max_gpu_elems = Helpers.get_max_feasible_elements()

    println("\n--- Starting Optimization ---")
    println("Log File: $log_filename")
    println("Results Folder: $RESULTS_DIR")
    println("Phase 1: Nominal ($nominal_iterations iters) - Maintain active elements ≈ $initial_target_count")
    println("Phase 2: Growth ($growth_steps iters) - Ramp active elements to $final_target_active")
    println("Phase 3: Annealing")

    while keep_running
        iter_start_time = time()
        status_msg = "Nominal"
        
        current_target_active = 0
        phase_refinement_needed = false
        
        # ----------------------------------------------------------------------
        # PHASE 1: NOMINAL ITERATIONS
        # ----------------------------------------------------------------------
        if iter <= nominal_iterations
            status_msg = "Nominal"
            is_annealing = false
            current_target_active = initial_target_count
            
            current_active = count(d -> d > 0.01, density)
            
            if iter > 5 && current_active < (initial_target_count * 0.8)
                phase_refinement_needed = true
            end

        # ----------------------------------------------------------------------
        # PHASE 2: GROWTH PHASE
        # ----------------------------------------------------------------------
        elseif iter <= (nominal_iterations + growth_steps)
            status_msg = "Growth"
            is_annealing = true 
            
            step_in_growth = iter - nominal_iterations
            progress = Float32(step_in_growth) / Float32(growth_steps)
            
            current_target_active = round(Int, initial_target_count + (final_target_active - initial_target_count) * progress)
            
            current_active = count(d -> d > 0.01, density)
            
            if current_active < (current_target_active * 0.95)
                phase_refinement_needed = true
            end

        # ----------------------------------------------------------------------
        # PHASE 3: FINAL ANNEALING
        # ----------------------------------------------------------------------
        else
            status_msg = "Annealing"
            is_annealing = true
            phase_refinement_needed = false
        end

        # ----------------------------------------------------------------------
        # REFINEMENT LOGIC (SHARED)
        # ----------------------------------------------------------------------
        if phase_refinement_needed
            n_total_current = length(density)
            current_active = count(d -> d > 0.01, density)
            active_ratio = max(0.001, current_active / n_total_current)
            
            est_total_needed = round(Int, current_target_active / active_ratio)
            
            if est_total_needed < max_gpu_elems
                println("\n>>> MESH UPDATE ($status_msg) [Iter $iter]")
                println("    Current Active: $current_active -> Target Active: $current_target_active")
                println("    Estimated New Total: $est_total_needed")
                
                nodes, elements, density, dims = MeshRefiner.refine_mesh_and_fields(
                    nodes, elements, density, dims, current_target_active, domain_bounds
                )
                
                GC.gc()
                
                nElem_x_new, nElem_y_new, nElem_z_new = dims[1]-1, dims[2]-1, dims[3]-1
                current_dx = domain_bounds.len_x / nElem_x_new
                current_dy = domain_bounds.len_y / nElem_y_new
                current_dz = domain_bounds.len_z / nElem_z_new
                
                config["geometry"]["nElem_x_computed"] = nElem_x_new
                config["geometry"]["nElem_y_computed"] = nElem_y_new
                config["geometry"]["nElem_z_computed"] = nElem_z_new
                config["geometry"]["dx_computed"] = current_dx
                config["geometry"]["dy_computed"] = current_dy
                config["geometry"]["dz_computed"] = current_dz
                
                geom = (
                    nElem_x = nElem_x_new, nElem_y = nElem_y_new, nElem_z = nElem_z_new,
                    dx = current_dx, dy = current_dy, dz = current_dz,
                    shapes_to_add = geom.shapes_to_add, shapes_to_remove = geom.shapes_to_remove,
                    actual_elem_count = size(elements, 1),
                    max_domain_dim = geom.max_domain_dim
                )

                nNodes = size(nodes, 1)
                ndof = nNodes * 3
                bc_indicator = get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
                F = zeros(Float32, ndof)
                apply_external_forces!(F, Vector{Any}(forces_data), nodes, elements)
                
                _, original_density, protected_elements_mask = 
                    initialize_density_field(nodes, elements, geom.shapes_to_add, geom.shapes_to_remove, config)
                
                U_full = zeros(Float32, ndof)
                TopologyOptimization.reset_filter_cache!()
                status_msg = "Refined"
            else
                println("\n>>> WARNING: GPU Memory Limit hit during refinement. Cap forced.")
            end
        end

        # ----------------------------------------------------------------------
        # SOLVER & UPDATE STEPS
        # ----------------------------------------------------------------------

        if iter > 1
            Threads.@threads for e in 1:size(elements, 1)
                if protected_elements_mask[e]
                    density[e] = original_density[e]
                end
            end
        end
        
        config["current_outer_iter"] = iter

        U_full = Solver.solve_system(
            nodes, elements, E, nu, bc_indicator, F;
            density=density, config=config, min_stiffness_threshold=min_density, prune_voids=true 
        )
        
        compliance = dot(F, U_full)
        strain_energy = 0.5 * compliance
        
        principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_dir_field =
            compute_stress_field(nodes, elements, U_full, E, nu, density)
        
        active_stress_indices = findall(d -> d > 0.1f0, density)
        avg_l1_stress = isempty(active_stress_indices) ? 0.0f0 : mean(view(l1_stress_norm_field, active_stress_indices))
        
        vol_total = length(density)
        active_non_soft = count(d -> d > min_density, density)
        vol_frac = sum(density) / vol_total
        
        res_tuple = update_density!(
            density, l1_stress_norm_field, protected_elements_mask,
            E, l1_stress_allowable, iter, nominal_iterations + growth_steps, 
            original_density, min_density, max_density_clamp,
            config, elements, is_annealing
        )
        max_change, filter_R, curr_threshold = res_tuple
        
        iter_time = time() - iter_start_time
        cur_dims_str = "$(config["geometry"]["nElem_x_computed"])x$(config["geometry"]["nElem_y_computed"])x$(config["geometry"]["nElem_z_computed"])"
        
        Diagnostics.write_iteration_log(
            log_filename, iter, cur_dims_str, vol_total, active_non_soft, 
            filter_R, curr_threshold, compliance, strain_energy, avg_l1_stress, vol_frac, max_change, 
            status_msg, iter_time
        )

        should_export = (iter == 1) || (iter % export_freq == 0) || status_msg == "Refined" || is_annealing
        if should_export
            export_iteration_results(
                iter, base_name, RESULTS_DIR, nodes, elements,
                U_full, F, bc_indicator, principal_field,
                vonmises_field, full_stress_voigt,
                l1_stress_norm_field, principal_dir_field, density, E, geom;
                iso_threshold=iso_threshold 
            )
        end
        
        if iter > (nominal_iterations + growth_steps)
            if max_change < convergence_threshold
                println("\n>>> CONVERGENCE ACHIEVED (Delta < $convergence_threshold)")
                keep_running = false
            end
             if iter > (nominal_iterations + growth_steps + 100)
                println("\n>>> MAX ANNEALING ITERATIONS REACHED")
                keep_running = false
             end
        end
        
        if nominal_iterations == 0; keep_running = false; end

        if CUDA.functional(); Helpers.clear_gpu_memory(); end
        iter += 1
        GC.gc() 
    end
    Diagnostics.log_status("Finished.")
end

end

using .HEXA

if length(ARGS) > 0 && isfile(ARGS[1])
    HEXA.run_main(ARGS[1])
else
    # Try finding config at PROJECT_ROOT/configs/default.yaml
    default_config = joinpath(Main.PROJECT_ROOT, "configs", "default.yaml")
    if isfile(default_config)
        HEXA.run_main(default_config)
    else
        println("\n!!! ERROR: No config file provided and default not found at: $default_config")
    end
end