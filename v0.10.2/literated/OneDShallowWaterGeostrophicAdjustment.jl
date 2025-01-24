using FourierFlows, CairoMakie, Printf, Random, JLD2
using LinearAlgebra: mul!, ldiv!

struct Params{T} <: AbstractParams
   ν :: T         # Hyperviscosity coefficient
  nν :: Int       # Order of the hyperviscous operator
   g :: T         # Gravitational acceleration
   H :: T         # Fluid depth
   f :: T         # Coriolis parameter
end
nothing #hide

struct Vars{Aphys, Atrans} <: AbstractVars
   u :: Aphys
   v :: Aphys
   η :: Aphys
  uh :: Atrans
  vh :: Atrans
  ηh :: Atrans
end
nothing #hide

"""
    Vars(grid)

Construct the `Vars` for 1D linear shallow water dynamics based on the dimensions of the `grid` arrays.
"""
function Vars(grid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T grid.nx u v η
  @devzeros Dev Complex{T} grid.nkr uh vh ηh

  return Vars(u, v, η, uh, vh, ηh)
end
nothing #hide

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Compute the nonlinear terms for 1D linear shallow water dynamics.
"""
function calcN!(N, sol, t, clock, vars, params, grid)
  @. vars.uh = sol[:, 1]
  @. vars.vh = sol[:, 2]
  @. vars.ηh = sol[:, 3]

  @. N[:, 1] =   params.f * vars.vh - im * grid.kr * params.g * vars.ηh    #  + f v - g ∂η/∂x
  @. N[:, 2] = - params.f * vars.uh                                        #  - f u
  @. N[:, 3] = - im * grid.kr * params.H * vars.uh                         #  - H ∂u/∂x

  dealias!(N, grid)

  return nothing
end
nothing #hide

"""
    Equation(params, grid)

Construct the equation: the linear part, in this case the hyperviscous dissipation,
and the nonlinear part, which is computed by `calcN!` function.
"""
function Equation(params, grid)
  T = eltype(grid)
  dev = grid.device

  L = zeros(dev, T, (grid.nkr, 3))
  D = @. - params.ν * grid.kr^(2*params.nν)

  L[:, 1] .= D # for u equation
  L[:, 2] .= D # for v equation
  L[:, 3] .= D # for η equation

  return FourierFlows.Equation(L, calcN!, grid)
end
nothing #hide

"""
    updatevars!(prob)

Update the variables in `prob.vars` using the solution in `prob.sol`.
"""
function updatevars!(prob)
  vars, grid, sol = prob.vars, prob.grid, prob.sol

  @. vars.uh = sol[:, 1]
  @. vars.vh = sol[:, 2]
  @. vars.ηh = sol[:, 3]

  ldiv!(vars.u, grid.rfftplan, deepcopy(sol[:, 1])) # use deepcopy() because irfft destroys its input
  ldiv!(vars.v, grid.rfftplan, deepcopy(sol[:, 2])) # use deepcopy() because irfft destroys its input
  ldiv!(vars.η, grid.rfftplan, deepcopy(sol[:, 3])) # use deepcopy() because irfft destroys its input

  return nothing
end
nothing #hide

"""
    set_uvη!(prob, u0, v0, η0)

Set the state variable `prob.sol` as the Fourier transforms of `u0`, `v0`, and `η0`
and update all variables in `prob.vars`.
"""
function set_uvη!(prob, u0, v0, η0)
  vars, grid, sol = prob.vars, prob.grid, prob.sol

  A = typeof(vars.u) # determine the type of vars.u

  # below, e.g., A(u0) converts u0 to the same type as vars expects
  # (useful when u0 is a CPU array but grid.device is GPU)
  mul!(vars.uh, grid.rfftplan, A(u0))
  mul!(vars.vh, grid.rfftplan, A(v0))
  mul!(vars.ηh, grid.rfftplan, A(η0))

  @. sol[:, 1] = vars.uh
  @. sol[:, 2] = vars.vh
  @. sol[:, 3] = vars.ηh

  updatevars!(prob)

  return nothing
end
nothing #hide

dev = CPU()    # Device (CPU/GPU)
nothing # hide

     nx = 512            # grid resolution
stepper = "FilteredRK4"  # timestepper
     dt = 20.0           # timestep (s)
 nsteps = 320            # total number of time-steps
nothing # hide

Lx = 500e3      # Domain length (m)
g  = 9.8        # Gravitational acceleration (m s⁻²)
H  = 200.0      # Fluid depth (m)
f  = 1e-2       # Coriolis parameter (s⁻¹)
ν  = 100.0      # Viscosity (m² s⁻¹)
nν = 1          # Viscosity order (nν = 1 means Laplacian ∇²)
nothing # hide

    grid = OneDGrid(dev; nx, Lx)
  params = Params(ν, nν, g, H, f)
    vars = Vars(grid)
equation = Equation(params, grid)

    prob = FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
nothing #hide

gaussian_width = 6e3
gaussian_amplitude = 3.0
gaussian_bump = @. gaussian_amplitude * exp( - grid.x^2 / (2*gaussian_width^2) )

fig = Figure(resolution = (600, 260))
ax =  Axis(fig[1, 1];
           xlabel = "x [km]",
           ylabel = "η [m]",
           title = "A gaussian bump with half-width ≈ " * string(gaussian_width/1e3) * " km",
           limits = ((-Lx/2e3, Lx/2e3), nothing))

lines!(ax, grid.x/1e3, gaussian_bump;    # divide with 1e3 to convert m -> km
       color = (:black, 0.7),
       linewidth = 2)

save("gaussian_bump.svg", fig); nothing # hide

mask = @. 1/4 * (1 + tanh( -(grid.x - 100e3) / 10e3)) * (1 + tanh( (grid.x + 100e3) / 10e3))

noise_amplitude = 0.1 # the amplitude of the noise for η(x, t=0) (m)
η_noise = noise_amplitude * Random.randn(size(grid.x))
@. η_noise *= mask    # mask the noise

fig = Figure(resolution = (600, 520))

kwargs = (xlabel = "x [km]", limits = ((-Lx/2e3, Lx/2e3), nothing))

ax1 =  Axis(fig[1, 1]; ylabel = "η [m]", title = "small-scale noise", kwargs...)

ax2 =  Axis(fig[2, 1]; ylabel = "mask", kwargs...)

lines!(ax1, grid.x/1e3, η_noise;      # divide with 1e3 to convert m -> km
       color = (:black, 0.7),
       linewidth = 3)

lines!(ax2, grid.x/1e3, mask;         # divide with 1e3 to convert m -> km
       color = (:gray, 0.7),
       linewidth = 2)

save("noise-mask.svg", fig); nothing # hide

η0 = @. gaussian_bump + η_noise
u0 = zeros(grid.nx)
v0 = zeros(grid.nx)

set_uvη!(prob, u0, v0, η0)

fig = Figure(resolution = (600, 260))

ax =  Axis(fig[1, 1];
           xlabel = "x [km]",
           ylabel = "η [m]",
           title = "initial surface elevation, η(x, t=0)",
           limits = ((-Lx/2e3, Lx/2e3), nothing))

lines!(ax, grid.x/1e3, η0;    # divide with 1e3 to convert m -> km
       color = (:black, 0.7),
       linewidth = 2)

save("initial_eta.svg", fig); nothing # hide

filepath = "."
filename = joinpath(filepath, "linear_swe.jld2")

get_sol(prob) = prob.sol

out = Output(prob, filename, (:sol, get_sol))

saveproblem(out)

for j = 0:nsteps
  updatevars!(prob)
  stepforward!(prob)
  saveoutput(out)
end

using JLD2

file = jldopen(out.path)
iterations = parse.(Int, keys(file["snapshots/t"]))

nx = file["grid/nx"]
 x = file["grid/x"]

 n = Observable(1)

u = @lift irfft(file[string("snapshots/sol/", iterations[$n])][:, 1], nx)
v = @lift irfft(file[string("snapshots/sol/", iterations[$n])][:, 2], nx)
η = @lift irfft(file[string("snapshots/sol/", iterations[$n])][:, 3], nx)

toptitle = @lift "t = " * @sprintf("%.1f", file[string("snapshots/t/", iterations[$n])]/60) * " min"

fig = Figure(resolution = (600, 800))

kwargs_η = (xlabel = "x [km]", limits = ((-Lx/2e3, Lx/2e3), nothing))
kwargs_uv = (xlabel = "x [km]", limits = ((-Lx/2e3, Lx/2e3), (-0.3, 0.3)))

ax_η =  Axis(fig[2, 1]; ylabel = "η [m]", title = toptitle, kwargs_η...)

ax_u =  Axis(fig[3, 1]; ylabel = "u [m s⁻¹]", kwargs_uv...)

ax_v =  Axis(fig[4, 1]; ylabel = "v [m s⁻¹]", kwargs_uv...)

Ld = @sprintf "%.2f" sqrt(g * H) / f /1e3     # divide with 1e3 to convert m -> km
title = "Deformation radius √(gh) / f = "*string(Ld)*" km"

fig[1, 1] = Label(fig, title, fontsize=24, tellwidth=false)

lines!(ax_η, grid.x/1e3, η; # divide with 1e3 to convert m -> km
       color = (:blue, 0.7))

lines!(ax_u, grid.x/1e3, u; # divide with 1e3 to convert m -> km
       color = (:red, 0.7))

lines!(ax_v, grid.x/1e3, v; # divide with 1e3 to convert m -> km
       color = (:green, 0.7))

frames = 1:length(iterations)

record(fig, "onedshallowwater.mp4", frames, framerate=18) do i
    n[] = i
end
nothing #hide

u_geostrophic = zeros(grid.nx)  # -g/f ∂η/∂y = 0
v_geostrophic = params.g / params.f * irfft(im * grid.kr .* vars.ηh, grid.nx)  #g/f ∂η/∂x

nothing # hide

fig = Figure(resolution = (600, 600))

kwargs = (xlabel = "x [km]", limits = ((-Lx/2e3, Lx/2e3), (-0.3, 0.3)))

ax_u =  Axis(fig[2, 1]; ylabel = "u [m s⁻¹]", kwargs...)

ax_v =  Axis(fig[3, 1]; ylabel = "v [m s⁻¹]", kwargs...)

fig[1, 1] = Label(fig, "Geostrophic balance", fontsize=24, tellwidth=false)

lines!(ax_u, grid.x/1e3, vars.u; # divide with 1e3 to convert m -> km
       label = "u",
       linewidth = 3,
       color = (:red, 0.7))

lines!(ax_u, grid.x/1e3, u_geostrophic; # divide with 1e3 to convert m -> km
       label = "- g/f ∂η/∂y",
       linewidth = 3,
       color = (:purple, 0.7))

axislegend(ax_u)

lines!(ax_v, grid.x/1e3, vars.v; # divide with 1e3 to convert m -> km
       label = "v",
       linewidth = 3,
       color = (:green, 0.7))

lines!(ax_v, grid.x/1e3, v_geostrophic; # divide with 1e3 to convert m -> km
       label = "g/f ∂η/∂x",
       linewidth = 3,
       color = (:purple, 0.7))

axislegend(ax_v)

save("geostrophic_balance.svg", fig); nothing # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

