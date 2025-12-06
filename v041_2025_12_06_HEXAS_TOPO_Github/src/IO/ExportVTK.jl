// # FILE: .\src\IO\ExportVTK.jl
module ExportVTK 

using Printf 

export export_mesh, export_solution 

function export_mesh(nodes::Matrix{Float32}, 
                     elements::Matrix{Int}; 
                     bc_indicator=nothing, 
                     filename::String="mesh_output.vtu") 
     
    if !endswith(lowercase(filename), ".vtk") && !endswith(lowercase(filename), ".vtu") 
        filename *= ".vtk" 
    end 
    # Mesh export logic simplified for this example as solution export is primary
end 

function export_solution(nodes::Matrix{Float32}, 
                         elements::Matrix{Int}, 
                         U_full::Vector{Float32}, 
                         F::Vector{Float32}, 
                         bc_indicator::Matrix{Float32}, 
                         principal_field::Matrix{Float32}, 
                         vonmises_field::Vector{Float32}, 
                         full_stress_voigt::Matrix{Float32}, 
                         l1_stress_norm_field::Vector{Float32},
                         principal_dir_field::Matrix{Float32}; 
                         density::Union{Vector{Float32}, Nothing}=nothing,
                         scale::Float32=Float32(1.0), 
                         filename::String="solution_output.vtu") 

    function sanitize_data(data) 
        data = replace(data, NaN => Float32(0.0), Inf => Float32(0.0), -Inf => Float32(0.0)) 
        max_val = maximum(abs.(data)) 
        if max_val > Float32(1.0e10) 
            @warn "Very large values detected (max abs: $max_val). Clamping." 
            return clamp.(data, Float32(-1.0e10), Float32(1.0e10)) 
        end 
        return data 
    end 
      
    U_full = sanitize_data(U_full) 
    F = sanitize_data(F) 
    nodes = sanitize_data(nodes) 
    principal_field = sanitize_data(principal_field) 
    principal_dir_field = sanitize_data(principal_dir_field) 
    vonmises_field = sanitize_data(vonmises_field) 
    full_stress_voigt = sanitize_data(full_stress_voigt) 
    l1_stress_norm_field = sanitize_data(l1_stress_norm_field) 

    nNodes = size(nodes, 1) 
    nElem  = size(elements, 1) 

    valid_elements = Int[] 
    for e = 1:nElem 
        push!(valid_elements, e) 
    end 
    nElem_valid = length(valid_elements) 

    # Prepare Displacement and Force vectors
    displacement = zeros(Float32, nNodes, 3) 
    forces = zeros(Float32, nNodes, 3) 
      
    for i in 1:nNodes 
        base_idx = 3*(i-1) 
        if base_idx + 3 <= length(U_full) 
            displacement[i, 1] = U_full[base_idx + 1] 
            displacement[i, 2] = U_full[base_idx + 2] 
            displacement[i, 3] = U_full[base_idx + 3] 
        end 
        if base_idx + 3 <= length(F) 
            forces[i, 1] = F[base_idx + 1] 
            forces[i, 2] = F[base_idx + 2] 
            forces[i, 3] = F[base_idx + 3] 
        end 
    end 
    disp_mag = sqrt.(sum(displacement.^2, dims=2))[:,1]     

    # Scaling for visualization
    max_disp = maximum(abs.(displacement)) 
    if max_disp > 0 
        max_dim = maximum([maximum(nodes[:,d]) - minimum(nodes[:,d]) for d in 1:3]) 
        if scale * max_disp > max_dim * 5 
            @warn "Scale factor causes very large deformation => auto reducing." 
            scale = Float32(0.5) * max_dim / max_disp 
        end 
    end 

    deformed_nodes = copy(nodes) 
    for i in 1:nNodes 
        deformed_nodes[i,1] += scale*displacement[i,1] 
        deformed_nodes[i,2] += scale*displacement[i,2] 
        deformed_nodes[i,3] += scale*displacement[i,3] 
    end 
    deformed_nodes = sanitize_data(deformed_nodes) 

    # Slicing active data
    principal_field_valid = principal_field[:, valid_elements] 
    principal_dir_valid   = principal_dir_field[:, valid_elements]
    vonmises_field_valid = vonmises_field[valid_elements] 
    l1_stress_norm_field_valid = l1_stress_norm_field[valid_elements] 
    full_stress_voigt_valid = full_stress_voigt[:, valid_elements] 

    if endswith(lowercase(filename), ".vtu"); combined_filename = filename[1:end-4] * ".vtk"
    elseif !endswith(lowercase(filename), ".vtk"); combined_filename = filename * ".vtk"
    else; combined_filename = filename; end

    try 
        open(combined_filename, "w") do file 
            write(file, "# vtk DataFile Version 3.0\n") 
            write(file, "HEXA FEM Solution (Single Precision Binary)\n") 
            write(file, "BINARY\n") 
            write(file, "DATASET UNSTRUCTURED_GRID\n") 
              
            # --- MODIFICATION: Enforce Float32 casting for single precision output ---
            write(file, "POINTS $(nNodes) float\n") 
            coords_flat = vec(deformed_nodes') # Transpose to get x,y,z linear
            write(file, hton.(Float32.(coords_flat))) 
              
            write(file, "\nCELLS $(nElem_valid) $(nElem_valid * 9)\n") 
            cell_data = Vector{Int32}(undef, nElem_valid * 9)
            idx = 1
            for e in valid_elements
                cell_data[idx] = Int32(8)
                idx += 1
                for j in 1:8
                    cell_data[idx] = Int32(elements[e, j] - 1)
                    idx += 1
                end
            end
            write(file, hton.(cell_data)) 
              
            write(file, "\nCELL_TYPES $(nElem_valid)\n") 
            cell_types = fill(Int32(12), nElem_valid) 
            write(file, hton.(cell_types)) 
              
            write(file, "\nPOINT_DATA $(nNodes)\n") 
              
            write(file, "VECTORS Displacement float\n") 
            disp_flat = vec(displacement')
            write(file, hton.(Float32.(disp_flat))) 
              
            write(file, "\nSCALARS Displacement_Magnitude float 1\n") 
            write(file, "LOOKUP_TABLE default\n") 
            write(file, hton.(Float32.(disp_mag))) 
              
            write(file, "\nVECTORS Force float\n") 
            force_flat = vec(forces')
            write(file, hton.(Float32.(force_flat))) 
              
            if size(bc_indicator, 1) == nNodes 
                 write(file, "\nVECTORS BC_Indicator float\n") 
                 bc_flat = vec(bc_indicator')
                 write(file, hton.(Float32.(bc_flat))) 
            end 
              
            write(file, "\nCELL_DATA $(nElem_valid)\n") 
              
            write(file, "SCALARS Von_Mises_Stress float 1\n") 
            write(file, "LOOKUP_TABLE default\n") 
            write(file, hton.(Float32.(vonmises_field_valid))) 
              
            write(file, "\nSCALARS l1_stress_norm float 1\n") 
            write(file, "LOOKUP_TABLE default\n") 
            write(file, hton.(Float32.(l1_stress_norm_field_valid))) 
              
            write(file, "\nVECTORS Principal_Stress_Values float\n") 
            principal_flat = vec(principal_field_valid)
            write(file, hton.(Float32.(principal_flat))) 

            write(file, "\nVECTORS Principal_Stress_Dir1 float\n") 
            pdir_flat = vec(principal_dir_valid)
            write(file, hton.(Float32.(pdir_flat))) 
              
            stress_names = ["Stress_XX", "Stress_YY", "Stress_ZZ", "Stress_XY", "Stress_YZ", "Stress_XZ"] 
            for idx in 1:6 
                write(file, "\nSCALARS $(stress_names[idx]) float 1\n") 
                write(file, "LOOKUP_TABLE default\n") 
                # row slicing
                stress_component = full_stress_voigt_valid[idx, :]
                write(file, hton.(Float32.(stress_component))) 
            end 

            if density !== nothing
                write(file, "\nSCALARS Element_Density float 1\n")
                write(file, "LOOKUP_TABLE default\n")
                density_valid = density[valid_elements]
                write(file, hton.(Float32.(density_valid)))
            end
        end 
    catch e 
        @error "Failed to save combined VTK file: $e" 
    end 
    return nothing 
end 

end