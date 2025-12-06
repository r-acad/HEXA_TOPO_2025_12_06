// # FILE: .\src\Solvers\GPUSolver.jl
module GPUSolver

using LinearAlgebra, Printf
using CUDA
using ..Element

export solve_system_gpu

# --- ANSI COLOR CONSTANTS ---
const C_RESET  = "\e[0m"
const C_BOLD   = "\e[1m"
const C_CYAN   = "\e[36m"
const C_GREEN  = "\e[32m"
const C_YELLOW = "\e[33m"
const C_RED    = "\e[31m"

function print_section_header(title::String, outer_iter::Any="?")
    width = 80
    println("\n" * C_CYAN * "="^width * C_RESET)
    
    # Format Title with Outer Iteration info
    full_title = "$title [Topo Opt Iter: $outer_iter]"
    
    padding = max(0, (width - length(full_title) - 2) ÷ 2)
    println(" "^padding * C_BOLD * full_title * C_RESET)
    println(C_CYAN * "="^width * C_RESET)
end

function get_free_dofs(bc_indicator::Matrix{T}) where T
    nNodes = size(bc_indicator, 1)
    ndof   = nNodes * 3
    constrained = falses(ndof)
    @inbounds for i in 1:nNodes
        if bc_indicator[i,1] > 0; constrained[3*(i-1)+1] = true; end
        if bc_indicator[i,2] > 0; constrained[3*(i-1)+2] = true; end
        if bc_indicator[i,3] > 0; constrained[3*(i-1)+3] = true; end
    end
    return findall(!, constrained)
end

function setup_matrix_free_operator(nodes::Matrix{T}, elements::Matrix{Int}, 
                                   E::T, nu::T, density::Vector{T}, 
                                   min_stiffness_threshold::T) where T
    
    setup_start = time()
    nElem = size(elements, 1)
    
    active_mask = density .>= min_stiffness_threshold
    active_indices = findall(active_mask)
    nActive = length(active_indices)
    
    if nActive == 0
        error(C_RED * "❌ No active elements found." * C_RESET)
    end

    n1 = nodes[elements[1,1], :]
    n2 = nodes[elements[1,2], :] 
    n4 = nodes[elements[1,4], :] 
    n5 = nodes[elements[1,5], :] 
    
    dx = norm(n2 - n1)
    dy = norm(n4 - n1)
    dz = norm(n5 - n1)
    
    Ke_base = Element.get_canonical_stiffness(dx, dy, dz, nu)
    
    active_elements = elements[active_indices, :]
    element_factors = E .* density[active_indices]
    
    return active_elements, Ke_base, element_factors, active_indices, setup_start
end

function matvec_kernel!(y::CuDeviceArray{T}, x::CuDeviceArray{T}, 
                        elements::CuDeviceArray{Int32}, 
                        Ke::CuDeviceArray{T},
                        factors::CuDeviceArray{T},
                        nActive::Int) where T
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nActive
        factor = factors[e]
        @inbounds n1, n2, n3, n4 = elements[e,1], elements[e,2], elements[e,3], elements[e,4]
        @inbounds n5, n6, n7, n8 = elements[e,5], elements[e,6], elements[e,7], elements[e,8]
        
        @inbounds begin
            x1 = x[3*(n1-1)+1]; x2 = x[3*(n1-1)+2]; x3 = x[3*(n1-1)+3]
            x4 = x[3*(n2-1)+1]; x5 = x[3*(n2-1)+2]; x6 = x[3*(n2-1)+3]
            x7 = x[3*(n3-1)+1]; x8 = x[3*(n3-1)+2]; x9 = x[3*(n3-1)+3]
            x10 = x[3*(n4-1)+1]; x11 = x[3*(n4-1)+2]; x12 = x[3*(n4-1)+3]
            x13 = x[3*(n5-1)+1]; x14 = x[3*(n5-1)+2]; x15 = x[3*(n5-1)+3]
            x16 = x[3*(n6-1)+1]; x17 = x[3*(n6-1)+2]; x18 = x[3*(n6-1)+3]
            x19 = x[3*(n7-1)+1]; x20 = x[3*(n7-1)+2]; x21 = x[3*(n7-1)+3]
            x22 = x[3*(n8-1)+1]; x23 = x[3*(n8-1)+2]; x24 = x[3*(n8-1)+3]
        end
        
        @inbounds for i in 1:24
            y_i = T(0.0)
            y_i += Ke[i, 1] * x1 + Ke[i, 2] * x2 + Ke[i, 3] * x3 + Ke[i, 4] * x4
            y_i += Ke[i, 5] * x5 + Ke[i, 6] * x6 + Ke[i, 7] * x7 + Ke[i, 8] * x8
            y_i += Ke[i, 9] * x9 + Ke[i, 10] * x10 + Ke[i, 11] * x11 + Ke[i, 12] * x12
            y_i += Ke[i, 13] * x13 + Ke[i, 14] * x14 + Ke[i, 15] * x15 + Ke[i, 16] * x16
            y_i += Ke[i, 17] * x17 + Ke[i, 18] * x18 + Ke[i, 19] * x19 + Ke[i, 20] * x20
            y_i += Ke[i, 21] * x21 + Ke[i, 22] * x22 + Ke[i, 23] * x23 + Ke[i, 24] * x24
            
            node_idx = (i - 1) ÷ 3 + 1
            dof_local = (i - 1) % 3 + 1
            node = (node_idx == 1 ? n1 : node_idx == 2 ? n2 : node_idx == 3 ? n3 : 
                    node_idx == 4 ? n4 : node_idx == 5 ? n5 : node_idx == 6 ? n6 : 
                    node_idx == 7 ? n7 : n8)
            global_dof = 3 * (node - 1) + dof_local
            CUDA.atomic_add!(pointer(y, global_dof), factor * y_i)
        end
    end
    return nothing
end

function apply_matrix_free_operator!(y::CuVector{T}, x::CuVector{T},
                                     elements_gpu::CuMatrix{Int32},
                                     Ke_gpu::CuMatrix{T},
                                     factors_gpu::CuVector{T},
                                     nActive::Int) where T
    fill!(y, T(0.0))
    threads = 256
    blocks = cld(nActive, threads)
    @cuda threads=threads blocks=blocks matvec_kernel!(y, x, elements_gpu, Ke_gpu, factors_gpu, nActive)
    CUDA.synchronize()
end

function expand_kernel!(x_full::CuDeviceArray{T}, x_free::CuDeviceArray{T}, 
                        free_to_full::CuDeviceArray{Int32}, n_free::Int) where T
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds x_full[free_to_full[idx]] = x_free[idx]
    end
    return nothing
end

function contract_kernel!(x_free::CuDeviceArray{T}, x_full::CuDeviceArray{T},
                          free_to_full::CuDeviceArray{Int32}, n_free::Int) where T
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_free
        @inbounds x_free[idx] = x_full[free_to_full[idx]]
    end
    return nothing
end

function jacobi_precond_kernel!(z::CuDeviceArray{T}, r::CuDeviceArray{T}, 
                                M_inv::CuDeviceArray{T}, n::Int) where T
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds z[idx] = r[idx] * M_inv[idx]
    end
    return nothing
end

function gpu_matrix_free_cg_solve(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                                  bc_indicator::Matrix{T}, f::Vector{T},
                                  density::Vector{T};
                                  max_iter=40000, tol=1e-6,
                                  shift_factor::T=Float32(1.0e-6),
                                  min_stiffness_threshold::T=Float32(1.0e-3),
                                  config::Dict=Dict()) where T
    CUDA.allowscalar(false)
    outer_iter = get(config, "current_outer_iter", "?")
    
    active_elements, Ke_base, element_factors, active_indices, setup_start = setup_matrix_free_operator(
        nodes, elements, E, nu, density, min_stiffness_threshold
    )
    
    nActive = length(active_indices)
    ndof = size(nodes, 1) * 3
    free_dofs = get_free_dofs(bc_indicator)
    n_free = length(free_dofs)
    
    print_section_header("GPU SOLVER - PCG (JACOBI)", outer_iter)
    @printf("  Free DOFs:             %12d\n", n_free)
    @printf("  Active Elements:       %12d\n", nActive)
    @printf("  Setup Time:            %12.3f s\n", time() - setup_start)

    free_to_full = Int32.(free_dofs)
    
    diag_full = zeros(T, ndof)
    @inbounds for t_idx in 1:nActive
        e = active_indices[t_idx]
        factor = E * density[e]
        conn = view(active_elements, t_idx, :)
        for i in 1:8
            node = conn[i]
            local_base = 3 * (i - 1)
            for dof in 1:3
                global_dof = 3 * (node - 1) + dof
                local_dof = local_base + dof
                diag_full[global_dof] += Ke_base[local_dof, local_dof] * factor
            end
        end
    end
    
    diag_free = diag_full[free_dofs]
    max_diag = maximum(abs.(diag_free))
    shift = shift_factor * max_diag
    M_inv_cpu = 1.0f0 ./ (diag_free .+ shift)
    
    @printf("  Diagonal shift:        %.4e\n", shift)
    println()

    elements_gpu = CuArray{Int32}(active_elements)
    Ke_gpu = CuArray(Ke_base)
    factors_gpu = CuArray(element_factors)
    free_to_full_gpu = CuArray(free_to_full)
    M_inv_gpu = CuArray(M_inv_cpu) 
    
    b_gpu = CuVector(f[free_dofs])
    x_gpu = CUDA.zeros(T, n_free)
    
    norm_b = norm(b_gpu)
    if norm_b == 0; return zeros(T, ndof); end

    r_gpu = CUDA.zeros(T, n_free)
    z_gpu = CUDA.zeros(T, n_free)
    p_gpu = CUDA.zeros(T, n_free)
    Ap_free_gpu = CUDA.zeros(T, n_free)
    x_full_gpu = CUDA.zeros(T, ndof)
    Ap_full_gpu = CUDA.zeros(T, ndof)
    
    threads_map = 256
    blocks_map = cld(n_free, threads_map)
    
    r_gpu .= b_gpu
    @cuda threads=threads_map blocks=blocks_map jacobi_precond_kernel!(z_gpu, r_gpu, M_inv_gpu, n_free)
    CUDA.synchronize()
    
    p_gpu .= z_gpu
    rz_old = dot(r_gpu, z_gpu)
    cg_start = time()
    converged = false
    
    println("  Starting PCG iterations...")
    # --- MODIFICATION: Output columns with reduced precision/width to save space ---
    @printf("\n  %s%s%8s %12s %12s %10s%s\n", 
            C_BOLD, C_CYAN, "CG Iter", "Res Norm", "Rel Res", "Time(s)", C_RESET)
    @printf("  %s%s\n", C_CYAN, "-"^56)
    
    for iter in 1:max_iter
        fill!(x_full_gpu, T(0.0))
        @cuda threads=threads_map blocks=blocks_map expand_kernel!(x_full_gpu, p_gpu, free_to_full_gpu, n_free)
        apply_matrix_free_operator!(Ap_full_gpu, x_full_gpu, elements_gpu, Ke_gpu, factors_gpu, nActive)
        @cuda threads=threads_map blocks=blocks_map contract_kernel!(Ap_free_gpu, Ap_full_gpu, free_to_full_gpu, n_free)
        CUDA.synchronize()
        
        Ap_free_gpu .+= shift .* p_gpu
        denom = dot(p_gpu, Ap_free_gpu)
        if abs(denom) < 1e-20; break; end
        alpha = rz_old / denom
        
        x_gpu .+= alpha .* p_gpu
        r_gpu .-= alpha .* Ap_free_gpu
        
        res_norm_sq = dot(r_gpu, r_gpu)
        res_norm = sqrt(res_norm_sq)
        rel_res = res_norm / norm_b
        
        # --- MODIFICATION: Log values with 4 digits max ---
        if (iter == 1) || (iter % 1000 == 0) || (rel_res < tol)
            color_code = (rel_res < tol) ? C_GREEN : C_RESET
            @printf("  %s%8d %12.4e %12.4e %10.3f%s\n", 
                    color_code, iter, res_norm, rel_res, time() - cg_start, C_RESET)
        end
        
        if rel_res < tol
            converged = true
            break
        end
        
        @cuda threads=threads_map blocks=blocks_map jacobi_precond_kernel!(z_gpu, r_gpu, M_inv_gpu, n_free)
        CUDA.synchronize()
        
        rz_new = dot(r_gpu, z_gpu)
        beta = rz_new / rz_old
        p_gpu .= z_gpu .+ beta .* p_gpu
        rz_old = rz_new
    end
    
    println(C_CYAN * "-"^56 * C_RESET)
    if converged
        println("  " * C_GREEN * "✓ CONVERGED" * C_RESET)
    else
        println("  " * C_RED * "⚠️ DID NOT CONVERGE" * C_RESET)
    end
    println(C_CYAN * "="^80 * C_RESET)
    
    x_full = zeros(T, ndof)
    x_full[free_dofs] = Array(x_gpu)
    return x_full
end

function solve_system_gpu(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, f::Vector{T},
                          density::Vector{T};
                          max_iter=1000, tol=1e-6, 
                          method=:native, solver=:cg, use_precond=true,
                          shift_factor::T=Float32(1.0e-6),
                          min_stiffness_threshold::T=Float32(1.0e-3),
                          config::Dict=Dict()) where T
    if !CUDA.functional(); error("CUDA not found."); end
    
    return gpu_matrix_free_cg_solve(nodes, elements, E, nu, bc_indicator, f, density,
                                    max_iter=max_iter, tol=tol, shift_factor=shift_factor,
                                    min_stiffness_threshold=min_stiffness_threshold,
                                    config=config)
end

end