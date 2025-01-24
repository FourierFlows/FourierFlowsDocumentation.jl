using FourierFlows, Plots
using FFTW: rfft, irfft
using LinearAlgebra: mul!, ldiv!
using Printf
using Random

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
    Vars!(dev, grid)
Constructs Vars based on the dimensions of arrays of the `grid`.
"""
function Vars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T grid.nx u v η
  @devzeros Dev Complex{T} grid.nkr uh vh ηh
  return Vars(u, v, η, uh, vh, ηh)
end
nothing #hide

"""
    calcN!(N, sol, t, clock, vars, params, grid)
The function that computes the nonlinear terms for our problem.
"""
function calcN!(N, sol, t, clock, vars, params, grid)
  @. vars.uh = sol[:, 1]
  @. vars.vh = sol[:, 2]
  @. vars.ηh = sol[:, 3]
  rhsu = @.   params.f * vars.vh - im * grid.kr * params.g * vars.ηh    #  + f v - g ∂η/∂x
  rhsv = @. - params.f * vars.uh                                        #  - f u
  rhsη = @. - im * grid.kr * params.H * vars.uh                         #  - H ∂u/∂x
  N[:, 1] .= rhsu
  N[:, 2] .= rhsv
  N[:, 3] .= rhsη
  dealias!(N, grid, grid.kralias)
  return nothing
end
nothing #hide

"""
    Equation!(prob)
Construct the equation: the linear part, in this case the hyperviscous dissipation,
and the nonlinear part, which is computed by `caclN!` function.
"""
function Equation(dev, params, grid::AbstractGrid)
  T = eltype(grid)
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
Sets the state variable `prob.sol` as the Fourier transforms of `u0`, `v0`, and `η0`
and update all variables in `prob.vars`.
"""
function set_uvη!(prob, u0, v0, η0)
  vars, grid, sol = prob.vars, prob.grid, prob.sol

  A = typeof(vars.u) # determine the type of vars.u

  mul!(vars.uh, grid.rfftplan, A(u0)) # A(u0) converts u0 to the same type as vars expects (useful if u0 is a CPU array while working on the GPU)
  mul!(vars.vh, grid.rfftplan, A(v0)) # A(v0) converts u0 to the same type as vars expects (useful if v0 is a CPU array while working on the GPU)
  mul!(vars.ηh, grid.rfftplan, A(η0)) # A(η0) converts u0 to the same type as vars expects (useful if η0 is a CPU array while working on the GPU)

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

grid = OneDGrid(dev, nx, Lx)
params = Params(ν, nν, g, H, f)
vars = Vars(dev, grid)
equation = Equation(dev, params, grid)

prob = FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
nothing #hide

gaussian_width = 6e3
gaussian_amplitude = 3.0
gaussian_bump = @. gaussian_amplitude * exp( - grid.x^2 / (2*gaussian_width^2) )

plot(grid.x / 1e3, gaussian_bump,    # divide with 1e3 to convert m -> km
     color = :black,
    legend = false,
 linewidth = 2,
     alpha = 0.7,
     xlims = (-Lx/2e3, Lx/2e3),
    xlabel = "x [km]",
    ylabel = "η [m]",
     title = "A gaussian bump with half-width ≈ "*string(gaussian_width/1e3)*" km",
      size = (600, 260))

mask = @. 1/4 * (1 + tanh( -(grid.x-100e3)/10e3 )) * (1 + tanh( (grid.x+100e3)/10e3 ))

noise_amplitude = 0.1 # the amplitude of the noise for η(x,t=0) (m)
η_noise = noise_amplitude * Random.randn(size(grid.x))
@. η_noise *= mask    # mask the noise

plot_noise = plot(grid.x / 1e3, η_noise,    # divide with 1e3 to convert m -> km
                 color = :black,
                legend = :false,
             linewidth = [3 2],
                 alpha = 0.7,
                 xlims = (-Lx/2e3, Lx/2e3),
                 ylims = (-0.3, 0.3),
                xlabel = "x [km]",
                ylabel = "η [m]")

plot_mask = plot(grid.x / 1e3, mask,    # divide with 1e3 to convert m -> km
                 color = :gray,
                legend = :false,
             linewidth = [3 2],
                 alpha = 0.7,
                 xlims = (-Lx/2e3, Lx/2e3),
                xlabel = "x [km]",
                ylabel = "mask")

title = plot(title = "Small-scale noise", grid = false, showaxis = false, bottom_margin = -20Plots.px)

plot(title, plot_noise, plot_mask,
           layout = @layout([A{0.01h}; [B; C]]),
             size = (600, 400))

η0 = @. gaussian_bump + η_noise
u0 = zeros(grid.nx)
v0 = zeros(grid.nx)

set_uvη!(prob, u0, v0, η0)

plot(grid.x / 1e3, η0,    # divide with 1e3 to convert m -> km
     color = :black,
    legend = false,
 linewidth = 2,
     alpha = 0.7,
     xlims = (-Lx/2e3, Lx/2e3),
    xlabel = "x [km]",
    ylabel = "η [m]",
     title = "initial surface elevation, η(x, t=0)",
      size = (600, 260))

function plot_output(prob)
  plot_η = plot(grid.x / 1e3, vars.η,    # divide with 1e3 to convert m -> km
                 color = :blue,
                legend = false,
             linewidth = 2,
                 alpha = 0.7,
                 xlims = (-Lx/2e3, Lx/2e3),
                xlabel = "x [km]",
                ylabel = "η [m]")

  plot_u = plot(grid.x / 1e3, vars.u,    # divide with 1e3 to convert m -> km
                 color = :red,
                legend = false,
             linewidth = 2,
                 alpha = 0.7,
                 xlims = (-Lx/2e3, Lx/2e3),
                 ylims = (-0.3, 0.3),
                xlabel = "x [km]",
                ylabel = "u [m s⁻¹]")

  plot_v = plot(grid.x / 1e3, vars.v,    # divide with 1e3 to convert m -> km
                 color = :green,
                legend = false,
             linewidth = 2,
                 alpha = 0.7,
                 xlims = (-Lx/2e3, Lx/2e3),
                 ylims = (-0.3, 0.3),
                xlabel = "x [km]",
                ylabel = "v [m s⁻¹]")

  Ld = @sprintf "%.2f" sqrt(g*H)/f /1e3  # divide with 1e3 to convert m -> km
  plottitle = "Deformation radius √(gh) / f = "*string(Ld)*" km"

  title = plot(title = plottitle, grid = false, showaxis = false, bottom_margin = -30Plots.px)

  return plot(title, plot_η, plot_u, plot_v,
           layout = @layout([A{0.01h}; [B; C; D]]),
             size = (600, 800))
end
nothing # hide

p = plot_output(prob)

anim = @animate for j=0:nsteps
  updatevars!(prob)

  p[2][1][:y] = vars.η    # updates the plot for η
  p[2][:title] = "t = "*@sprintf("%.1f", prob.clock.t / 60 )*" min" # updates time in the title
  p[3][1][:y] = vars.u    # updates the plot for u
  p[4][1][:y] = vars.v    # updates the plot for v

  stepforward!(prob)
end

mp4(anim, "onedshallowwater.mp4", fps=18)

u_geostrophic = zeros(grid.nx)  # -g/f ∂η/∂y = 0
v_geostrophic = params.g/params.f * irfft(im * grid.kr .* vars.ηh, grid.nx)  #g/f ∂η/∂x

plot_u = plot(grid.x / 1e3, [vars.u u_geostrophic],    # divide with 1e3 to convert m -> km
                 color = [:red :purple],
                labels = ["u" "- g/f ∂η/∂y"],
             linewidth = [3 2],
                 alpha = 0.7,
                 xlims = (-Lx/2e3, Lx/2e3),
                 ylims = (-0.3, 0.3),
                xlabel = "x [km]",
                ylabel = "u [m s⁻¹]")

plot_v = plot(grid.x / 1e3, [vars.v v_geostrophic],    # divide with 1e3 to convert m -> km
                 color = [:green :purple],
                labels = ["v" "g/f ∂η/∂x"],
             linewidth = [3 2],
                 alpha = 0.7,
                 xlims = (-Lx/2e3, Lx/2e3),
                 ylims = (-0.3, 0.3),
                xlabel = "x [km]",
                ylabel = "v [m s⁻¹]")

title = plot(title = "Geostrophic balance", grid = false, showaxis = false, bottom_margin = -20Plots.px)

plot(title, plot_u, plot_v,
           layout = @layout([A{0.01h}; [B; C]]),
             size = (600, 400))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

