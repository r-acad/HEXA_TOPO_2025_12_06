import Pkg

# Define the target file path
const TARGET_FILE = joinpath(@__DIR__, "Project.toml")

println(">>> Generating Project.toml from current environment...")

# 1. Gather Dependencies
deps = Pkg.dependencies()
# Sort them alphabetically for a clean file
sorted_deps = sort(collect(deps), by=x->x[2].name)

# 2. Prepare the content in memory
buffer = IOBuffer()

# --- HEADER ---
println(buffer, "name = \"HEXA_TopOpt\"")
println(buffer, "uuid = \"a1b2c3d4-e5f6-4a5b-9c8d-7e6f5g4h3i2j\"")
println(buffer, "authors = [\"User\"]")
println(buffer, "version = \"1.0.0\"")
println(buffer, "")

# --- [deps] SECTION ---
println(buffer, "[deps]")
for (uuid, pkg) in sorted_deps
    if pkg.is_direct_dep
        # uuid is the key of the dictionary
        println(buffer, "$(pkg.name) = \"$uuid\"")
    end
end
println(buffer, "")

# --- [compat] SECTION ---
println(buffer, "[compat]")
println(buffer, "julia = \"1.9\"") # Base Julia version

for (uuid, pkg) in sorted_deps
    if pkg.is_direct_dep
        if pkg.version !== nothing
            # The "=" sign enforces EXACT version matching
            println(buffer, "$(pkg.name) = \"=$(pkg.version)\"")
        else
            println(buffer, "# Warning: Could not detect version for $(pkg.name)")
        end
    end
end

# 3. Write to file
new_content = String(take!(buffer))

# Create a backup if one exists
if isfile(TARGET_FILE)
    mv(TARGET_FILE, TARGET_FILE * ".bak", force=true)
    println(">>> Existing Project.toml backed up to Project.toml.bak")
end

open(TARGET_FILE, "w") do io
    write(io, new_content)
end

println("-"^60)
println(">>> SUCCESS: Project.toml has been written successfully.")
println(">>> Location: $TARGET_FILE")
println("-"^60)