module GPUHelmholtz

using CUDA
using LinearAlgebra
using Printf
using ..Element

export HelmholtzWorkspace, setup_helmholtz_workspace, apply_gpu_filter!


mutable struct HelmholtzWorkspace{T}
    is_initialized::Bool
    radius::T
    
    # Static Mesh Data (GPU)
    elements::CuVector{Int32} 
    Ae_base::CuMatrix{T}       
    inv_diag::CuVector{T}      
    
    # CG Vectors (GPU)
    r::CuVector{T}
    p::CuVector{T}
    z::CuVector{T}
    Ap::CuVector{T}
    x::CuVector{T} 
    b::CuVector{T} 
    
    # Dimensions
    nNodes::Int
    nElem::Int
    
    HelmholtzWorkspace{T}() where T = new{T}(false, T(0))
end

const GLOBAL_HELMHOLTZ_CACHE = HelmholtzWorkspace{Float32}()

# ----------------------------------------------------------------------
# CUDA KERNELS
# ----------------------------------------------------------------------

function compute_rhs_kernel!(b, density, elements, val_scale, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        val = density[e] * val_scale
        base_idx = (e - 1) * 8
        @inbounds for i in 1:8
            node = elements[base_idx + i]
            CUDA.atomic_add!(pointer(b, node), val)
        end
    end
    return nothing
end

function matvec_kernel!(y, x, elements, Ae, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        
        # Load local x
        x_loc_1 = x[elements[base_idx + 1]]
        x_loc_2 = x[elements[base_idx + 2]]
        x_loc_3 = x[elements[base_idx + 3]]
        x_loc_4 = x[elements[base_idx + 4]]
        x_loc_5 = x[elements[base_idx + 5]]
        x_loc_6 = x[elements[base_idx + 6]]
        x_loc_7 = x[elements[base_idx + 7]]
        x_loc_8 = x[elements[base_idx + 8]]
        
        # Multiply by local 8x8 matrix Ae
        @inbounds for r in 1:8
            val = Ae[r,1]*x_loc_1 + Ae[r,2]*x_loc_2 + Ae[r,3]*x_loc_3 + Ae[r,4]*x_loc_4 +
                  Ae[r,5]*x_loc_5 + Ae[r,6]*x_loc_6 + Ae[r,7]*x_loc_7 + Ae[r,8]*x_loc_8
            
            node = elements[base_idx + r]
            CUDA.atomic_add!(pointer(y, node), val)
        end
    end
    return nothing
end

function extract_solution_kernel!(filtered_density, x, elements, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        sum_val = 0.0f0
        @inbounds for i in 1:8
            sum_val += x[elements[base_idx + i]]
        end
        filtered_density[e] = sum_val / 8.0f0
    end
    return nothing
end

function compute_diagonal_kernel!(diag, elements, Ae_diag, nElem)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= nElem
        base_idx = (e - 1) * 8
        @inbounds for i in 1:8
            node = elements[base_idx + i]
            val = Ae_diag[i]
            CUDA.atomic_add!(pointer(diag, node), val)
        end
    end
    return nothing
end

# ----------------------------------------------------------------------
# SETUP & SOLVE
# ----------------------------------------------------------------------

function setup_helmholtz_workspace(elements_cpu::Matrix{Int}, 
                                   dx::T, dy::T, dz::T, radius::T) where T
    
    ws = GLOBAL_HELMHOLTZ_CACHE
    nElem = size(elements_cpu, 1)
    nNodes = maximum(elements_cpu)
    
    # Initialize only if parameters changed or first run
    if !ws.is_initialized || ws.nElem != nElem || abs(ws.radius - radius) > 1e-5
        
        # 1. Compute Local Matrix
        Ke, Me = Element.get_scalar_canonical_matrices(dx, dy, dz)
        Ae_cpu = (radius^2) .* Ke .+ Me
        
        # 2. Upload Geometry
        elements_flat = vec(elements_cpu') 
        ws.elements = CuArray(Int32.(elements_flat))
        ws.Ae_base = CuArray(Ae_cpu)
        
        # 3. Compute Diagonal Preconditioner
        diag_vec = CUDA.zeros(T, nNodes)
        Ae_diag_gpu = CuArray(diag(Ae_cpu))
        
        threads = 256
        blocks = cld(nElem, threads)
        @cuda threads=threads blocks=blocks compute_diagonal_kernel!(diag_vec, ws.elements, Ae_diag_gpu, nElem)
        
        ws.inv_diag = 1.0f0 ./ diag_vec
        
        # 4. Allocate CG Vectors
        ws.r  = CUDA.zeros(T, nNodes)
        ws.p  = CUDA.zeros(T, nNodes)
        ws.z  = CUDA.zeros(T, nNodes)
        ws.Ap = CUDA.zeros(T, nNodes)
        ws.x  = CUDA.zeros(T, nNodes)
        ws.b  = CUDA.zeros(T, nNodes)
        
        ws.nNodes = nNodes
        ws.nElem = nElem
        ws.radius = radius
        ws.is_initialized = true
        
        println("  [GPU Filter] Workspace initialized for $(nElem) elements.")
    end
    return ws
end

function apply_gpu_filter!(density_cpu::Vector{T}, nElem_x, nElem_y, nElem_z, dx, dy, dz, radius, elements_cpu) where T
    
    # 1. Setup / Retrieve Workspace
    ws = setup_helmholtz_workspace(elements_cpu, T(dx), T(dy), T(dz), T(radius))
    
    density_gpu = CuArray(density_cpu) 
    filtered_gpu = CUDA.zeros(T, ws.nElem)
    
    threads = 256
    blocks = cld(ws.nElem, threads)
    
    # 2. Compute RHS (b = M * rho)
    fill!(ws.b, 0.0f0)
    elem_vol = dx * dy * dz
    val_scale = elem_vol / 8.0f0
    @cuda threads=threads blocks=blocks compute_rhs_kernel!(ws.b, density_gpu, ws.elements, val_scale, ws.nElem)
    
    norm_b = norm(ws.b)
    if norm_b == 0.0f0
        return density_cpu # Field is empty
    end

    # 3. Conjugate Gradient Loop (Matrix-Free)
    fill!(ws.x, 0.0f0) 
    ws.r .= ws.b
    ws.z .= ws.r .* ws.inv_diag # Jacobi Preconditioner
    ws.p .= ws.z
    
    rho_old = dot(ws.r, ws.z)
    
    tol = 1e-4
    max_iter = 200 
    final_rel_res = 0.0f0
    final_iter = 0

    for iter in 1:max_iter
        final_iter = iter
        
        fill!(ws.Ap, 0.0f0)
        @cuda threads=threads blocks=blocks matvec_kernel!(ws.Ap, ws.p, ws.elements, ws.Ae_base, ws.nElem)
        
        alpha = rho_old / dot(ws.p, ws.Ap)
        
        ws.x .+= alpha .* ws.p
        ws.r .-= alpha .* ws.Ap
        
        # Check Convergence
        norm_r = norm(ws.r)
        final_rel_res = norm_r / norm_b
        
        if final_rel_res < tol
            break
        end
        
        ws.z .= ws.r .* ws.inv_diag 
        
        rho_new = dot(ws.r, ws.z)
        beta = rho_new / rho_old
        ws.p .= ws.z .+ beta .* ws.p
        
        rho_old = rho_new
    end
    
    # --- MODIFIED: Print Convergence Info ---
    # We print on a single line to avoid cluttering the log
    @printf("  [GPU Filter] Radius: %.3f | Converged: %s | Iters: %3d | RelRes: %.1e\n", 
            radius, (final_rel_res < tol ? "YES" : "NO "), final_iter, final_rel_res)

    # 4. Extract solution back to elements
    @cuda threads=threads blocks=blocks extract_solution_kernel!(filtered_gpu, ws.x, ws.elements, ws.nElem)
    
    return Array(filtered_gpu)
end

end