var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#FourierFlows.jl-Documentation-1",
    "page": "Home",
    "title": "FourierFlows.jl Documentation",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Overview-1",
    "page": "Home",
    "title": "Overview",
    "category": "section",
    "text": "FourierFlows provides solvers for partial differential equations on doubly-periodic domains using Fourier-based pseudospectral methods. A central intent of the software\'s design is also to provide a framework for writing new, fast solvers for new physical problems. The code is written in Julia."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "FourierFlows is a registered package and so can be installed via Pkg.add.Pkg.add(\"FourierFlows\")For now, this package supports Julia 0.6. Support for version 0.7 is on its way."
},

{
    "location": "index.html#Basic-Notation-1",
    "page": "Home",
    "title": "Basic Notation",
    "category": "section",
    "text": "The code solves partial differential equations of the general form:partial_t u = mathcalLu + mathcalN(u) We decompose the right hand side of the above in a linear part (mathcalLu) and a nonlinear part (mathcalN(u)). The time steppers treat the linear and nonlinear parts differently.The coefficients for the linear operator mathcalL are stored in array LC. The term mathcalN(u) is computed for by calling the function calcN!."
},

{
    "location": "index.html#Source-code-organization-1",
    "page": "Home",
    "title": "Source code organization",
    "category": "section",
    "text": "The code is divided along conceptual lines into problem-agnostic and problem-specific components. Files that contain problem-agnostic parts of the code are stored in /src. Files in /src define the domain, \'AbstractTypes\' that supertype problem-specific types, and time-stepper types and routines. Problem-specific modules are stores in /src/physics.Here\'s an overview of the code structure:/src/\nFourierFlows.jl\nDefines supertyping AbstractParams, AbstractGrid, etc.\nDefines a Problem type to organize the grid, vars, params,   equation, and timestepper into a single structure.\nIncludes all sources files and physics files.\ntimesteppers.jl: defines modules and stepforward! routines for   various time-steppers. Current implemented time-steppers are:\nForward Euler\n3rd-order Adams-Bashforth (AB3)\n4th-order Runge-Kutta (RK4)\n4th-order Runge-Kutta Exponential Time Differencing (ETDRK4)\n4th-order Dual Runge-Kutta (DualRK4)\n4th-order Dual Runge-Kutta Exponential Time Differencing (DualETDRK4)\nFor each time-stepper exists also a \"filtered\" version that filters   out high-wavenumber spectral components of the solution. The Dual   time-steppers evolve a state variable that comprises both of real valued   and complex valued fields.\nphysics/\ntwodturb.jl: Defines a TwoDTurb module that provides a       solver for the two-dimensional vorticity equation.\nbarotropicqg.jl: Defines a BarotropicQG module that provides       several solvers for the barotropic QG model that permit beta,       topography, beta + topography, and forcing.\nkuramotosivashinsky.jl: Defines a KuramotoSivashinsky module that       solves the Kuramoto-Sivashinsky.\nverticallyfourierboussinesq.jl: Defines a VerticallyFourierBoussinesq module that       solves the two-mode truncation of the Fourier series thin-layer approximation to the hydrostatic Boussinesq equations.\nverticallycosinerboussinesq.jl: Defines a VerticallyCosineBoussinesq module that       solves the two-mode truncation of the Sin/Cos series thin-layer approximation to the hydrostatic Boussinesq equations.\ntraceradvdiff.jl: Defines a TracerAdvDiff module that       provides a solver for a two-dimensional and periodic tracer       field in a given 2D flow (u, w), which can be an arbitrary       function of x, z, and t."
},

{
    "location": "index.html#Writing-fast-solvers-1",
    "page": "Home",
    "title": "Writing fast solvers",
    "category": "section",
    "text": "The performance-intensive part of the code involves just two functions: the time-stepping scheme stepforward!, and the function calcN! that calculates the nonlinear part of the given equation\'s right-hand side. Optimization of these two functions for a given problem will produce the fastest possible code."
},

{
    "location": "index.html#Examples-1",
    "page": "Home",
    "title": "Examples",
    "category": "section",
    "text": "examples/twodturb/McWilliams.jl: A script that simulates decaying two-dimensional turbulence reproducing the results of the paper by\nMcWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. J. Fluid Mech., 146, 21-43\nexamples/barotropicqg/decayingbetaturb.jl: An example script that simulates decaying quasi-geostrophic flow on a beta-plane demonstrating zonation.\nexamples/barotropicqg/ACConelayer.jl: A script that simulates barotropic quasi-geostrophic flow above topography reproducing the results of the paper by\nConstantinou, N. C. (2018). A barotropic model of eddy saturation. J. Phys. Oceanogr., 48 (2), 397-411"
},

{
    "location": "index.html#Tutorials-1",
    "page": "Home",
    "title": "Tutorials",
    "category": "section",
    "text": "Pages = [\n    \"modules/twodturb.md\",\n    \"modules/barotropicqg.md\"\n    \"modules/kuramotosivashinsky.md\"\n    \"modules/traceradvdiff.md\"\n        ]\nDepth = 1"
},

{
    "location": "index.html#Future-work-1",
    "page": "Home",
    "title": "Future work",
    "category": "section",
    "text": "The code is in the chaotic stage of development. A main goal for the future is to permit the use of shared memory parallelism in the compute-intensive routines (shared-memory parallelism provided already by FFTW/MKLFFT, but is not yet native to Julia for things like element-wise matrix multiplication, addition, and assignment). This feature may possibly be enabled by Intel Lab\'s ParallelAccelerator package."
},

{
    "location": "index.html#Developers-1",
    "page": "Home",
    "title": "Developers",
    "category": "section",
    "text": "FourierFlows is currently being developed by Gregory L. Wagner and Navid C. Constantinou."
},

{
    "location": "index.html#DocStrings-1",
    "page": "Home",
    "title": "DocStrings",
    "category": "section",
    "text": "Pages = [\n    \"modules/twodturb.md\",\n    \"modules/barotropicqg.md\",\n    \"modules/boussinesq.md\",\n    \"modules/kuramotosivashinsky.md\",\n    \"modules/traceradvdiff.md\",\n    \"man/docstrings.md\",\n    ]\nDepth = 2"
},

{
    "location": "index.html#Index-1",
    "page": "Home",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\n    \"modules/twodturb.md\",\n    \"modules/barotropicqg.md\",\n    \"modules/boussinesq.md\",\n    \"modules/kuramotosivashinsky.md\",\n    \"modules/traceradvdiff.md\",\n    \"man/docstrings.md\",\n    ]"
},

{
    "location": "modules/twodturb.html#",
    "page": "TwoDTurb Module",
    "title": "TwoDTurb Module",
    "category": "page",
    "text": ""
},

{
    "location": "modules/twodturb.html#TwoDTurb-Module-1",
    "page": "TwoDTurb Module",
    "title": "TwoDTurb Module",
    "category": "section",
    "text": "newcommandJmathsfJ"
},

{
    "location": "modules/twodturb.html#Basic-Equations-1",
    "page": "TwoDTurb Module",
    "title": "Basic Equations",
    "category": "section",
    "text": "This module solves two-dimensional incompressible turbulence. The flow is given through a streamfunction psi as (uupsilon) = (-partial_ypsi partial_xpsi). The dynamical variable used here is the component of the vorticity of the flow normal to the plane of motion, q=partial_x upsilon- partial_y u = nabla^2psi. The equation solved by the module is:partial_t q + J(psi q) = underbrace-leftmu(-1)^n_mu nabla^2n_mu\n+nu(-1)^n_nu nabla^2n_nuright q_textrmdissipation + f where J(a b) = (partial_x a)(partial_y b)-(partial_y a)(partial_x b). On the right hand side, f(xyt) is forcing, mu is hypoviscosity, and nu is hyperviscosity. Plain old linear drag corresponds to n_mu=0, while normal viscosity corresponds to n_nu=1."
},

{
    "location": "modules/twodturb.html#Implementation-1",
    "page": "TwoDTurb Module",
    "title": "Implementation",
    "category": "section",
    "text": "The equation is time-stepped forward in Fourier space:partial_t widehatq = - widehatJ(psi q) -leftmu k^2n_mu\n+nu k^2n_nuright widehatq  + widehatf In doing so the Jacobian is computed in the conservative form: J(fg) = partial_y  (partial_x f) g -partial_x (partial_y f) g.Thus:mathcalL = -mu k^-2n_mu - nu k^2n_nu mathcalN(widehatq) = - mathrmik_x mathrmFFT(u q)-\n	mathrmik_y mathrmFFT(upsilon q) "
},

{
    "location": "modules/barotropicqg.html#",
    "page": "BarotropicQG Module",
    "title": "BarotropicQG Module",
    "category": "page",
    "text": ""
},

{
    "location": "modules/barotropicqg.html#BarotropicQG-Module-1",
    "page": "BarotropicQG Module",
    "title": "BarotropicQG Module",
    "category": "section",
    "text": "newcommandJmathsfJ"
},

{
    "location": "modules/barotropicqg.html#Basic-Equations-1",
    "page": "BarotropicQG Module",
    "title": "Basic Equations",
    "category": "section",
    "text": "This module solves the quasi-geostrophic barotropic vorticity equation on a beta-plane of variable fluid depth H-h(xy). The flow is obtained through a streamfunction psi as (u upsilon) = (-partial_ypsi partial_xpsi). All flow fields can be obtained from the quasi-geostrophic potential vorticity (QGPV). Here the QGPV isunderbracef_0 + beta y_textplanetary PV + underbrace(partial_x upsilon\n	- partial_y u)_textrelative vorticity +\n	underbracefracf_0 hH_texttopographic PVThe dynamical variable is the component of the vorticity of the flow normal to the plane of motion, zetaequiv partial_x upsilon- partial_y u = nabla^2psi. Also, we denote the topographic PV with etaequiv f_0 hH. Thus, the equation solved by the module is:partial_t zeta + J(psi underbracezeta + eta_equiv q) +\nbetapartial_xpsi = underbrace-leftmu + nu(-1)^n_nu nabla^2n_nu\nright zeta _textrmdissipation + f where J(a b) = (partial_x a)(partial_y b)-(partial_y a)(partial_x b). On the right hand side, f(xyt) is forcing, mu is linear drag, and nu is hyperviscosity. Plain old viscosity corresponds to n_nu=1. The sum of relative vorticity and topographic PV is denoted with qequivzeta+eta."
},

{
    "location": "modules/barotropicqg.html#Implementation-1",
    "page": "BarotropicQG Module",
    "title": "Implementation",
    "category": "section",
    "text": "The equation is time-stepped forward in Fourier space:partial_t widehatzeta = - widehatJ(psi q) +betafracmathrmik_xk^2widehatzeta -left(mu\n+nu k^2n_nuright) widehatzeta  + widehatf In doing so the Jacobian is computed in the conservative form: J(fg) = partial_y  (partial_x f) g -partial_x (partial_y f) g.Thus:mathcalL = betafracmathrmik_xk^2 - mu - nu k^2n_nu mathcalN(widehatzeta) = - mathrmik_x mathrmFFT(u q)-\n	mathrmik_y mathrmFFT(upsilon q) "
},

{
    "location": "modules/boussinesq.html#",
    "page": "Thin-layer Boussinesq modules",
    "title": "Thin-layer Boussinesq modules",
    "category": "page",
    "text": ""
},

{
    "location": "modules/boussinesq.html#Thin-layer-Boussinesq-modules-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Thin-layer Boussinesq modules",
    "category": "section",
    "text": "newcommandbcdotboldsymbol cdot\nnewcommandbnablaboldsymbol nabla\nnewcommandpnablabnabla_ perp\n\nnewcommandcom \nnewcommandper \n\nnewcommandbuboldsymbol u\nnewcommandbUboldsymbol U\nnewcommandbuuboldsymbolu\nnewcommandbbb\nnewcommandppp\nnewcommandwww\nnewcommanduuu\nnewcommandvupsilon\nnewcommandvvupsilon\nnewcommandzzetazeta\nnewcommandoomegaomega\nnewcommandboomegaboldsymboloomega\n\nnewcommandbxhwidehatboldsymbolx\nnewcommandbyhwidehatboldsymboly\nnewcommandbzhwidehatboldsymbolz\nnewcommandiimathrmi\nnewcommandeemathrme\nnewcommandccmathrmcc\nnewcommandJmathsfJ\n\nnewcommandppartialThese modules solve various thin-layer approximations to the hydrostatic Boussinesq equations. A thin-layer approximation is one that is appropriate for dynamics with small aspect ratios, or small vertical scales and large horizontal scales. Thin layer approximations include the shallow-water system, layered system, and spectral approximations that apply a Fourier or Sin/Cos eigenfunction expansion in the vertical coordinate to the Boussinesq equations, and truncate the expansion at just two or three modes. Approximations of this last flavor are described here.The three-dimensional rotating, stratified, hydrostatic Boussinesq equations arep_tbuu + left ( buu bcdot bnabla right ) buu + f bzh times buu + bnabla pp = D^buu com \np_z pp = bb com \np_tbb + ww N^2 = D^bb com \nbnabla bcdot buu = 0 comwhere bu = (u v w) is the three-dimensional velocity, b is buoyancy, p is pressure, N^2 is the buoyancy frequency (constant), and f is the rotation or Coriolis frequency. The operators D^buu and D^bb are arbitrary dissipation that we define only after projecting onto vertical Fourier or Sin/Cos modes. Taking the curl of the horizontal momentum equation yields an evolution equation for vertical vorticity, zzeta = p_x vv - p_y uu:p_tzzeta + buu bcdot bnabla zzeta - left (f bzh + boomega right )\n    bcdot bnabla ww = D^zzeta per"
},

{
    "location": "modules/boussinesq.html#Vertically-Fourier-Boussinesq-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Vertically Fourier Boussinesq",
    "category": "section",
    "text": "The vertically-Fourier Boussinesq module solves the Boussinesq system obtained by expanding the hydrostatic Boussinesq equations in a Fourier series. The horizontal velocity uu, for example, is expanded withuu(x y z t) mapsto U(x y t) + ee^ii m z u(x y t) + ee^-ii m z u^*(x y t) comThe other variables vv, bb, pp, zzeta, and boomega are expanded identically. The barotropic horizontal velocity is V and the barotropic vertical vorticity is Z = p_x V - p_y U. The barotropic vorticity obeysp_t Z + J left ( Psi Z right )\n    + bnabla bcdot left ( bu zeta^* right ) + ii m pnabla bcdot left ( bu w^* right ) + cc\n    = D_0 Z comwhere cc denotes the complex conjugate and contraction with pnabla = -p_y bxh + p_x byh gives the vertical component of the curl.The baroclinic components obeyp_t u - f v + p_x p = - J left ( Psi u right ) - bu bcdot bnabla U + D_1 u com \np_t v + f u + p_y p = - J left ( Psi v right ) - bu bcdot bnabla V + D_1 v com \np_t p - tfracN^2m w = - J left ( Psi p right ) + D_1 p perThe dissipation operators are definedD_0 = nu_0 (-1)^n_0 nabla^2n_0 + mu_0 (-1)^m_0 nabla^2m_0 com \nD_1 = nu_1 (-1)^n_1 nabla^2n_1 + mu_1 (-1)^m_1 nabla^2m_1where U is the barotropic velocity and u is the amplitude of the first baroclinic mode with periodic vertical structure mathrme^mathrmi m z."
},

{
    "location": "modules/boussinesq.html#Implementation-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Implementation",
    "category": "section",
    "text": "Coming soon."
},

{
    "location": "modules/boussinesq.html#Vertically-Cosine-Boussinesq-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Vertically Cosine Boussinesq",
    "category": "section",
    "text": "The vertically-Cosine Boussinesq module solves the Boussinesq system obtained by expanding the hydrostatic Boussinesq equations in a Sin/Cos series. The horizontal velocity, for example, becomesuu(x y z t) mapsto U(x y t) + cos(mz) u(x y t) perThe horizontal velocity vv, pressure pp, and vertical vorticity zzeta are also expanded in cos(mz), where Z = p_x V - p_y U denotes the barotropic component of the vertical vorticity. The vertical velocity ww and buoyancy bb are expanded with sin(mz)."
},

{
    "location": "modules/boussinesq.html#Basic-governing-equations-1",
    "page": "Thin-layer Boussinesq modules",
    "title": "Basic governing equations",
    "category": "section",
    "text": "Projecting the vertical vorticity equation onto Sin/Cos modes an equation for the evolution of Z,p_t Z + J left ( Psi Z right )\n    + tfrac12 bnabla bcdot left ( bu zeta right ) + tfracm2 pnabla bcdot left ( bu w right )\n    = D_0 Z comwhere J(a b) = (p_x a)(p_y b) - (p_y a)(p_x b) is the Jacobian operator, contraction with pnabla = -p_y bxh + p_x byh gives the vertical component of the curl, and Psi is the barotropic streamfunction defined so thatbU = -p_yPsi bxh + p_xPsi byh qquad textand qquad Z = nabla^2 Psi perThe baroclinic components obeyp_t u - f v + p_x p = - J left ( Psi u right ) - bu bcdot bnabla U + D_1u com \np_t v + f u + p_y p = - J left ( Psi v right ) - bu bcdot bnabla V + D_1v com \np_t p - tfracN^2m w = - J left ( Psi p right ) + D_1p perThe dissipation operators are definedD_0 = nu_0 (-1)^n_0 nabla^2n_0 + mu_0 (-1)^m_0 nabla^2m_0 com \nD_1 = nu_1 (-1)^n_1 nabla^2n_1 + mu_1 (-1)^m_1 nabla^2m_1 comwhere 2n_0 and 2m_0 are the hyperviscous orders of the arbitrary barotropic dissipation operators with coefficients nu_0 and mu_0, while 2n_1 and 2m_1 are the orders of the baroclinic dissipation operators.A passive tracer in the Vertically Cosine Boussinesq system is assumed to satisfy a no-flux condition at the upper and lower boundaries, and thus expanded in cosine modes so thatc(x y z t) = C(x y t) + cos(mz) c(x y t) perThe barotropic and baroclinic passive tracer components then obeyp_t C + J(Psi C) + tfrac12 bnabla bcdot left ( bu c right ) =\n    kappa (-1)^n_kappa nabla^2n_kappa C com \np_t c + J(Psi c) + bu bcdot bnabla C = kappa (-1)^n_kappa nabla^2n_kappa c comwhere kappa and n_kappa are the tracer hyperdiffusivity and order of the hyperdiffusivity, respectively. The choice n_kappa = 1 corresponds to ordinary Fickian diffusivity."
},

{
    "location": "modules/boussinesq.html#Implementation-2",
    "page": "Thin-layer Boussinesq modules",
    "title": "Implementation",
    "category": "section",
    "text": "Coming soon."
},

{
    "location": "modules/kuramotosivashinsky.html#",
    "page": "Kuramoto-Sivashinsky Module",
    "title": "Kuramoto-Sivashinsky Module",
    "category": "page",
    "text": ""
},

{
    "location": "modules/kuramotosivashinsky.html#Kuramoto-Sivashinsky-Module-1",
    "page": "Kuramoto-Sivashinsky Module",
    "title": "Kuramoto-Sivashinsky Module",
    "category": "section",
    "text": ""
},

{
    "location": "modules/kuramotosivashinsky.html#Basic-Equations-1",
    "page": "Kuramoto-Sivashinsky Module",
    "title": "Basic Equations",
    "category": "section",
    "text": "This module solves the Kuramoto-Sivashinsky equation for u(xt):partial_t u + partial_x^4 u + partial_x^2 u + upartial_x u = 0 "
},

{
    "location": "modules/kuramotosivashinsky.html#Implementation-1",
    "page": "Kuramoto-Sivashinsky Module",
    "title": "Implementation",
    "category": "section",
    "text": "The equation is time-stepped forward in Fourier space:partial_t widehatu + k_x^4 widehatu - k_x^2 widehatu + widehat upartial_x u  = 0 Thus:mathcalL = k_x^2 - k_x^4 mathcalN(widehatu) = - mathrmFFT(u partial_x u) The function calcN! implements dealiasing to avoid energy piling up at the grid-scale."
},

{
    "location": "modules/traceradvdiff.html#",
    "page": "TracerAdvDiff Module",
    "title": "TracerAdvDiff Module",
    "category": "page",
    "text": ""
},

{
    "location": "modules/traceradvdiff.html#TracerAdvDiff-Module-1",
    "page": "TracerAdvDiff Module",
    "title": "TracerAdvDiff Module",
    "category": "section",
    "text": ""
},

{
    "location": "modules/traceradvdiff.html#Basic-Equations-1",
    "page": "TracerAdvDiff Module",
    "title": "Basic Equations",
    "category": "section",
    "text": "This module solves the advection diffusion equation for a passive tracer concentration c(xyt) in two-dimensions:partial_t c + boldsymbolu boldsymbolcdot boldsymbolnabla c = kappa nabla^2 c where mathbfu = (uv) is the two-dimensional advecting velocity and kappa is the diffusivity."
},

{
    "location": "modules/traceradvdiff.html#Implementation-1",
    "page": "TracerAdvDiff Module",
    "title": "Implementation",
    "category": "section",
    "text": "Coming soon."
},

{
    "location": "man/docstrings.html#",
    "page": "Functions exported from FourierFlows:",
    "title": "Functions exported from FourierFlows:",
    "category": "page",
    "text": ""
},

{
    "location": "man/docstrings.html#Base.resize!-Tuple{FourierFlows.AbstractDiagnostic,Int64}",
    "page": "Functions exported from FourierFlows:",
    "title": "Base.resize!",
    "category": "method",
    "text": "resize!(diag, newnum)\n\nResize the Diagnostic data and time arrays to length newnum.\n\n\n\n"
},

{
    "location": "man/docstrings.html#FourierFlows.increment!-Tuple{FourierFlows.AbstractDiagnostic}",
    "page": "Functions exported from FourierFlows:",
    "title": "FourierFlows.increment!",
    "category": "method",
    "text": "increment!(diag)\nincrement!(diags)\n\nIncrement the Diagnostic diag, or an array of Diagnostics diags.\n\n\n\n"
},

{
    "location": "man/docstrings.html#FourierFlows.savediagnostic-Tuple{FourierFlows.AbstractDiagnostic,String,String}",
    "page": "Functions exported from FourierFlows:",
    "title": "FourierFlows.savediagnostic",
    "category": "method",
    "text": "savediagnostic(diag, diagname)\n\nSave diagnostics to file, labeled by the string diagname.\n\n\n\n"
},

{
    "location": "man/docstrings.html#FourierFlows.saveoutput-Tuple{FourierFlows.Output}",
    "page": "Functions exported from FourierFlows:",
    "title": "FourierFlows.saveoutput",
    "category": "method",
    "text": "saveoutput(out)\n\nSave current output fields for file in out.filename.\n\n\n\n"
},

{
    "location": "man/docstrings.html#FourierFlows.saveproblem-Tuple{FourierFlows.AbstractProblem,String}",
    "page": "Functions exported from FourierFlows:",
    "title": "FourierFlows.saveproblem",
    "category": "method",
    "text": "saveproblem(prob, filename)\n\nSave certain aspects of a problem timestepper, grid, and params. Functions that are fields in params are not saved.\n\n\n\n"
},

{
    "location": "man/docstrings.html#FourierFlows.stepforward!-Tuple{FourierFlows.Problem,AbstractArray,Any}",
    "page": "Functions exported from FourierFlows:",
    "title": "FourierFlows.stepforward!",
    "category": "method",
    "text": "stepforward!(prob, diags, nsteps)\n\nStep forward prob for nsteps, incrementing diagnostics in the array diags along the way.\n\n\n\n"
},

{
    "location": "man/docstrings.html#FourierFlows.stepforward!-Tuple{FourierFlows.Problem,Any}",
    "page": "Functions exported from FourierFlows:",
    "title": "FourierFlows.stepforward!",
    "category": "method",
    "text": "stepforward!(prob, nsteps)\n\nStep forward prob for nsteps.\n\n\n\n"
},

{
    "location": "man/docstrings.html#FourierFlows.stepforward!-Tuple{FourierFlows.Problem}",
    "page": "Functions exported from FourierFlows:",
    "title": "FourierFlows.stepforward!",
    "category": "method",
    "text": "stepforward!(prob)\n\nStep forward the Problem prob for one timestep.\n\n\n\n"
},

{
    "location": "man/docstrings.html#FourierFlows.update!-Tuple{FourierFlows.AbstractDiagnostic}",
    "page": "Functions exported from FourierFlows:",
    "title": "FourierFlows.update!",
    "category": "method",
    "text": "update!(diag)\n\nUpdate diag with its current value.\n\n\n\n"
},

{
    "location": "man/docstrings.html#Functions-exported-from-FourierFlows:-1",
    "page": "Functions exported from FourierFlows:",
    "title": "Functions exported from FourierFlows:",
    "category": "section",
    "text": "Modules = [FourierFlows]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/docstrings.html#FourierFlows.ZeroDGrid",
    "page": "Functions exported from FourierFlows:",
    "title": "FourierFlows.ZeroDGrid",
    "category": "type",
    "text": "ZeroDGrid()\n\nConstructs a placeholder grid object for \"0D\" problems (in other words, systems of ODEs).\n\n\n\n"
},

{
    "location": "man/docstrings.html#Private-types-in-module-FourierFlows:-1",
    "page": "Functions exported from FourierFlows:",
    "title": "Private types in module FourierFlows:",
    "category": "section",
    "text": "Modules = [FourierFlows]\nPublic = false\nOrder = [:type]"
},

]}
