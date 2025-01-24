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
    "location": "index.html#Usage-1",
    "page": "Home",
    "title": "Usage",
    "category": "section",
    "text": "The code solves partial differential equations of the general form:partial_t u = mathcalLu + mathcalN(u) We decompose the right hand side of the above in a linear part (mathcalLu) and a nonlinear part (mathcalN(u)). The nonlinear part may include external forcing, e.g., mathcalN(u) = -upartial_x u + f."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "FourierFlows is a registered package and so can be installed via Pkg.add.Pkg.add(\"FourierFlows\")For now, this package supports Julia 0.6. Support for version 0.7 is on its way."
},

{
    "location": "index.html#Writing-fast-solvers-1",
    "page": "Home",
    "title": "Writing fast solvers",
    "category": "section",
    "text": "The performance-intensive part of the code involves just two functions: the time-stepping scheme stepforward!, and the function calcN! that calculates the nonlinear part of the given equation\'s right-hand side. Optimization of these two functions for a given problem will produce the fastest possible code."
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
    "location": "index.html#Cite-1",
    "page": "Home",
    "title": "Cite",
    "category": "section",
    "text": "The code is citable via zenodo. Please cite as:Gregory L. Wagner & Navid C. Constantinou. (2018). FourierFlows/FourierFlows.jl: FourierFlows v0.1.1 (Version v0.1.1). Zenodo.  http://doi.org/10.5281/zenodo.1302136"
},

{
    "location": "basics.html#",
    "page": "Code Basics",
    "title": "Code Basics",
    "category": "page",
    "text": ""
},

{
    "location": "basics.html#Code-Basics-1",
    "page": "Code Basics",
    "title": "Code Basics",
    "category": "section",
    "text": ""
},

{
    "location": "basics.html#Basic-Notation-1",
    "page": "Code Basics",
    "title": "Basic Notation",
    "category": "section",
    "text": "The code solves partial differential equations of the general form:partial_t u = mathcalLu + mathcalN(u) (Note: ODEs are special cases of the above. Thus the code also solves ODEs.)We decompose the right hand side of the above in a linear part (mathcalLu) and a nonlinear part (mathcalN(u)). The time steppers treat the linear and nonlinear parts differently.Boundary conditions in all spatial dimensions are periodic. That allows us to expand all variables using a Fourier decomposition. For example, a variable phi(x t) that depends in one spatial dimension is expanded as:phi(x t) = sum_k widehatphi(k t)e^mathrmi k x where wavenumbers k take the values tfrac2piL_x0pm 1pm 2dots. The equation is time-stepped forward in Fourier space. That way u becomes the array with all Fourier coefficients of the solution.The coefficients for the linear operator mathcalL are stored in an array called LC. The term mathcalN(u) is computed for by calling the function calcN!."
},

{
    "location": "basics.html#Abstract-SuperTypes-1",
    "page": "Code Basics",
    "title": "Abstract SuperTypes",
    "category": "section",
    "text": "The code is divided along conceptual lines into problem-agnostic and problem-specific components. Files that contain problem-agnostic parts of the code are stored in /src. Files in /src define the domain, \'AbstractTypes\' that supertype problem-specific types, and time-stepper types and routines. Problem-specific modules are stored in /src/physics.Below is a list of all Abstract Supertypes used by the code:AbstractGrid: Includes all variables that have to do with the grid, both in physical space as well as in wavenumber space. Currently implemented are: ZeroGrid for ODEs, OneGrid for PDEs with one spatial dimension, and TwoGrid for PDEs with two spatial dimensions. Grids are generic and work for any problem of that particular dimension.\nAbstractParams: Includes all parameters or functions related with the problem do not vary throughout the integration.\nAbstractVars: Includes all variables of the problem that change along the integration.\nAbstractEquation: Includes the array with the coefficients of the linear part of the equation, LC as well as function calcN! that computes the nonlinear part of the equation.\nAbstractState: Includes the solution sol at current time-step as well as the time-step dt, the time t, and step which counts the number of time-steps taken.\nAbstractTimeStepper: Includes all details for the time-stepper (e.g., dt, various coefficients, sol at intermediate time-step values). Time-steppers are generic and work for any problem.\nAbstractProblem: A super-supertype that includes all of the above. That is problem includes grid, vars, params, eqn, ts, state, and also t and step.Grids and time-steppers are generic and work for any problem of that particular dimension. State and Problem just gathers things together. Thus, to write a solver for a new physical problem you only need to prescribe params, vars, the coefficients of the linear part, LC, and function calcN!."
},

{
    "location": "basics.html#Source-code-organization-1",
    "page": "Code Basics",
    "title": "Source code organization",
    "category": "section",
    "text": "The code is divided along conceptual lines into problem-agnostic and problem-specific components. Files that contain problem-agnostic parts of the code are stored in /src. Files in /src define the domain, \'AbstractTypes\' that supertype problem-specific types, and time-stepper types and routines. Problem-specific modules are stores in /src/physics.Here\'s an overview of the code structure:/src/\nFourierFlows.jl\nDefines supertyping AbstractParams, AbstractGrid, etc.\nDefines a Problem type to organize the grid, vars, params, equation, and timestepper into a single structure.\nIncludes all sources files and physics files.\ntimesteppers.jl: defines modules and stepforward! routines for   various time-steppers. Current implemented time-steppers are:\nForward Euler\n3rd-order Adams-Bashforth (AB3)\n4th-order Runge-Kutta (RK4)\n4th-order Runge-Kutta Exponential Time Differencing (ETDRK4)\n4th-order Dual Runge-Kutta (DualRK4)\n4th-order Dual Runge-Kutta Exponential Time Differencing (DualETDRK4)\nFor each time-stepper exists also a \"filtered\" version that filters out high-wavenumber spectral components of the solution. The Dual time-steppers evolve a state variable that comprises both of real valued         and complex valued fields.\nphysics/\ntwodturb.jl: Defines a TwoDTurb module that provides a solver for the two-dimensional vorticity equation.\nbarotropicqg.jl: Defines a BarotropicQG module that provides several solvers for the barotropic QG model that permit beta, topography, beta + topography, and forcing.\nkuramotosivashinsky.jl: Defines a KuramotoSivashinsky module that solves the Kuramoto-Sivashinsky.\nverticallyfourierboussinesq.jl: Defines a VerticallyFourierBoussinesq module that solves the two-mode truncation of the Fourier series thin-layer approximation to the hydrostatic Boussinesq equations.\nverticallycosinerboussinesq.jl: Defines a VerticallyCosineBoussinesq module that solves the two-mode truncation of the Sin/Cos series thin-layer approximation to the hydrostatic Boussinesq equations.\ntraceradvdiff.jl: Defines a TracerAdvDiff module that provides a solver for a two-dimensional and periodic tracer field in a given 2D flow (u, w), which can be an arbitrary function of x, z, and t."
},

{
    "location": "basics.html#Basic-steps-for-solving-a-problem:-step-through-an-example-script-1",
    "page": "Code Basics",
    "title": "Basic steps for solving a problem: step through an example script",
    "category": "section",
    "text": "To illustrate the basic steps for solving a problem consider the 1D Kuramoto-Sivashinsky equation for u(x t):partial_t u + partial_x^4 u + partial_x^2 u + upartial_x u = 0 which in Fourier base reads:partial_t widehatu = underbrace(- k_x^4 + k_x^2) widehatu_mathcalLwidehatu\n+ underbracewidehat -upartial_x u _mathcalN(widehatu) The steps to construct an AbstractProblem for the above are:Construct an AbstractGrid; for this problem we use the OneGrid.\nConstruct an AbstractParams; for this problem params is be empty as there are no parameters in the equation. (Note that e.g., the domain size Lx and the number of gridpoints nx belong to the grid.)\nConstruct an AbstractVars; for this problem vars includes u, partial_x u, upartial_x u and their Fourier transforms widehatu, widehatpartial_x u, widehatupartial_xu.\nConstruct the equations by prescribing coefficients for the linear part as an array LC and a function calcN! that computes mathcalN(widehatu).\nConstruct the time-stepper which includes function stepforward! that time-steps the solution.\nConstruct the state and gather everything as an AbstractProblem.The example script found in  examples/kuramotosivashinsky/trefethenexample.jl demonstrates the above steps needed to construct an AbstractProblem. The prob is constructed by calling prob = InitialValueProblem(nx=nx, Lx=Lx, dt=dt, stepper=\"ETDRK4\"). Looking into the  InitialValueProblem function we can see the above steps:function InitialValueProblem(;\n     nx = 256,\n     Lx = 2π,\n     dt = 0.01,\nstepper = \"RK4\"\n)\n\ng  = OneDGrid(nx, Lx)\npr = Params()\nvs = Vars(g)\neq = Equation(pr, g)\nts = FourierFlows.autoconstructtimestepper(stepper, dt, eq.LC, g)\n\nFourierFlows.Problem(g, vs, pr, eq, ts)\nendThe OneDGrid function is called for the grid. Within grid the wavenumber array is constructed:i1 = 0:Int(nx/2)\ni2 = Int(-nx/2+1):-1\nk = Array{T}(2π/Lx*cat(1, i1, i2))\nkr = Array{T}(2π/Lx*cat(1, i1))For real-valued fields we use rfft and thus only positive wavenumbers are involved: array kr. E.g., for nx=8 and Lx=2π the wavenumber grids are: k = [0, 1, 2, 3, 4, -3, -2, -1] and kr = [0, 1, 2, 3, 4].The construction of the grids only works for even number of grid points. Moreover, since the code relies on the mathrmFFT algorithm, we suggest you use a power of 2 as the number of grid points, since then mathrmFFT is most efficient. Function Vars(g) initialize variables u, ux, and uux as real valued arrays of length nx and variables uh, uxh, and uuxh as complex valued arrays of length nkr = Int(nx/2+1) (the same length as kr). As a general convention variable names with h denote the Fourier transforms of the corresponding variable (h stands for \'hat\').The array LC is constructed by Equation functionfunction Equation(p, g)\n  LC = @. g.kr^2 - g.kr^4\n  FourierFlows.Equation(LC, calcN!)\nendAlso eq includes function calcN! which computes the nonlinear term mathcalN(widehatu):function calcN!(N, sol, t, s, v, p, g)\n  @. v.uh = sol\n  @. v.uxh = im*g.kr*sol\n  A_mul_B!(v.u, g.irfftplan, v.uh)\n  A_mul_B!(v.ux, g.irfftplan, v.uxh)\n  @. v.uux = v.u*v.ux\n  A_mul_B!(v.uuxh, g.rfftplan, v.uux)\n  @. N = -v.uuxh\n  dealias!(N, g)\n  nothing\nendThe time-stepper is constructed and stored as ts. Finally, all supertypes are gathered together as an AbstractProblem."
},

{
    "location": "basics.html#Tutorials-1",
    "page": "Code Basics",
    "title": "Tutorials",
    "category": "section",
    "text": "Pages = [\n    \"modules/kuramotosivashinsky.md\",\n    \"modules/twodturb.md\",\n    \"modules/barotropicqg.md\",\n    \"modules/traceradvdiff.md\"\n        ]\nDepth = 1"
},

{
    "location": "forcing.html#",
    "page": "Forcing",
    "title": "Forcing",
    "category": "page",
    "text": "newcommandsqrmboxsqr\nnewcommandsawmboxsaw\nnewcommandindmboxind\nnewcommandsgnmboxsgn\nnewcommanderfcmboxerfc\nnewcommanderfmboxerf\n\n An average\nnewcommandavg1mathrmavg 1 \n The right way to define new functions\nnewcommandsechmathoprm sechnolimits\nnewcommandcosechmathoprm cosechnolimits\n\n A nice definition\nnewcommanddefnstackrelmathrmdef=\n \n\nnewcommandol1overline1\n\n\n Various boldsymbols\nnewcommandbxboldsymbolx\nnewcommandbyboldsymboly\nnewcommandbqboldsymbolq\nnewcommandbpsiboldsymbolpsi\nnewcommandbuboldsymbolu\nnewcommandbGboldsymbolmathcalG\nnewcommandGmathcalG\nnewcommandbaboldsymbola\nnewcommandbbboldsymbolb\nnewcommandbcboldsymbolc\nnewcommandbvboldsymbolv\nnewcommandbkboldsymbolk\nnewcommandbXboldsymbolX\nnewcommandbrboldsymbolr\nnewcommandJmathsfJ\nnewcommandDmathsfD\nrenewcommandLmathsfL\nnewcommandsLmathsfL\nnewcommandGboldsymbolmathsfG\nnewcommandbAboldsymbol A\nnewcommandbUboldsymbol U\nnewcommandbEboldsymbol E\nnewcommandbJboldsymbol J\nnewcommandbXXboldsymbol mathcalX\nnewcommandbFFensuremath boldsymbol F\nnewcommandbFensuremath boldsymbol F^sharp\nnewcommandbLensuremath boldsymbol L\nnewcommandbIensuremath boldsymbol I\nnewcommandbNensuremath boldsymbol N\n\nnewcommandIensuremath mathsfI\nrenewcommandLensuremath mathsfL\nrenewcommandSensuremath mathsfS\n\nnewcommandbSigmaensuremath boldsymbol Sigma\nnewcommandkmaxk_mathrmmax\nnewcommandbnablaboldsymbolnabla\nnewcommandbcdotboldsymbolcdot\n\ndefiirm i\ndefddrm d\ndefeerm e\ndefDDrm D\n Cals here \n\n  Euler caligraphics \nnewcommandAmathscrA\n newcommandBmathscrB\nnewcommandBmathcalB\nnewcommandEmathscrE\nnewcommandFmathscrF\nnewcommandKmathscrK\nnewcommandNmathscrN\nnewcommandUmathscrU\nnewcommandLLmathscrL\nnewcommandMmathscrM\nnewcommandTmathscrT\ndeflalangle\ndefrarangle\ndeflaaleft langle\ndefraaright rangle\ndefEkmathrmEk\nnewcommandhzonh_mathrmzon\nnewcommandlaptriangle\nnewcommandppartial\nnewcommandhalf tfrac12\nnewcommandgradboldsymbol nabla\nnewcommandpdetextscpde\nnewcommandodetextscode\nnewcommandcctextsccc\nnewcommanddctextscdc\nnewcommanddbctextscdbc\nnewcommandbyutextscbyu\nnewcommandrhstextscrhs\nnewcommandlhstextsclhs\nnewcommandcom\nnewcommandper\nnewcommandzzeta\nnewcommandheta\nrenewcommand(left(\nrenewcommandleft\nrenewcommand)right)\nrenewcommandright\nnewcommandleftlangle\nrenewcommandrightrangle\nrenewcommandAmathcalA\nrenewcommandNmathcalN\nnewcommandCmathcalC\nnewcommandtransptextrmT\nnewcommandzhatwidehatmathbfz\n\nnewcommandbitvphantomdotW\nnewcommandsdb"
},

{
    "location": "forcing.html#Forcing-1",
    "page": "Forcing",
    "title": "Forcing",
    "category": "section",
    "text": "The code implements forcing in various modules (currently in TwoDTurb and BarotropicQG). Forcing can be either deterministic or stochastic (random). For deterministic forcing the implementation is straightforward; for stochastic forcing there are two main train of thoughts: Itô calculus and Stratonovich calculus.Both stochastic calculi give the same results. But once we decide to use one of the two calculi we have to remain consistent and use that calculus for everywhere. There is a lot of confusion and mostly the confusion stems from not using the same stochastic calculus consistently throughout the computation but rather interchanging between the two.FourierFlows uses Stratonovich calculus throughout the code. This choise was made because Stratonovich calculus works the same with both stochastic and deterministic forcing, i.e. with Stratonovich calculus we have the same chain rules for differentiation for stochastic functions as the chain rules we learn in normal-deterministic calculus). Therefore, the code written as is does not really \"care\" of what forcing the user implements.If you are interested in learning more regarding the two stochastic calculi and how they are numerically implemented then read on; otherwise you can skip this section of the documentation and go to the Module Tutorials."
},

{
    "location": "forcing.html#Stochastic-Differential-Equations-(SDEs)-1",
    "page": "Forcing",
    "title": "Stochastic Differential Equations (SDEs)",
    "category": "section",
    "text": "A differential equation in the form: \\begin{equation} 	\\frac{\\dd x}{\\dd t} = f(x)\\com\\quad x(t0)=0\\com \\end{equation} can also be written in an integral form: \\begin{equation} 	x(t) = \\int{t0}^{t} f(x(s))\\,\\dd s\\per \\end{equation} In a similar manner, a stochastic differential equation \\begin{equation} 	\\dd x = f(x)\\,\\dd t + g(x)\\,\\dd Wt\\com\\quad x(t0)=0\\com \\end{equation} with \\dd Wt$ a white-noise process, can be written in an integral form as: \\begin{equation} 	x(t) = \\int{t0}^{t} f(x(s))\\,\\dd s + \\int{t0}^{t} g(x(s))\\,\\dd W_s \\per \\end{equation} Of course now, the last integral is a stochastic integral and there is not a single straight-forward way of computing it –- there are a lot of different ways we can approximate it as a Riemannian sum and each of them leads to a different answer. The two most popular ways for computing such stochastic integrals are:colorGreentextItô int_t_0^t g(x(s))dd W_sapproxsum_j gleft(x(t_j)right)(W_j+1-W_j)com\ncolorMagentatextStratonovich int_t_0^t g(x(s))dd W_s approx sum_j gleft(xleft(half(t_j+t_j+1)right)right)(W_j+1-W_j)perBecause the white noise process is not continuous the two definitions do not converge to the same result; the two definitions give thoroughly different results. And to overcome that they come along with different chain rules, i.e., chain rules that are not necessarily the same as those in plain old calculus.An SDE can be written also in differential form. Because we cannot formally form dd Wdd t, since W is nowhere differentiable, we write an SDE in differential form as:colorGreentextItô dd x_t = f(x_t)dd t + g(x_t)dd W_tcom\ncolorMagentatextStratonovich dd x_t = f(x_t)dd t + g(x_t)circdd W_tperThe circle in g(x_t)circdd W_t is used to differentiate between Itô or Stratonovich calculus.A variable change y=G(x) is done as follows according to the two different calculi:colorGreentextItô dd y_t = fracdd Gdd xdd x_t + half g(x_t)^2 fracdd^2 Gdd x^2dd t =left fracdd Gdd xf(x_t) + half g(x_t)^2 fracdd^2 Gdd x^2rightdd t + fracdd Gdd xg(x_t)dd W_tcom\ncolorMagentatextStratonovich dd y_t  = fracdd Gdd xdd x_t =fracdd Gdd x f(x_t) dd t + fracdd Gdd xg(x_t)dd W_tperThe above are the so called stochastic chain rules. All derivatives of G are evaluated at x_t.It\'s easy to see that the extra drift-term in Itô\'s interpretation of the stochastic integral, i.e., colorGreenhalf g^2 dd^2Gdd x^2  is exactly equal to the ensemble mean of the Stratonovich stochastic integral. This is that case because, by construction, the Itô stochastic integral has zero ensemble mean since at every instant the noise is multiplied with g evaluated before the action of the noise occurs; g and dd W are uncorrelated and thus: \\begin{equation} {\\color{Green}\\laa g(xt)\\dd Wt \\raa =0}\\quad\\text{while}\\quad {\\color{Magenta}\\laa g(xt)\\circ\\dd Wt \\raa \\ne 0}\\per \\end{equation} The above is demonstrated by evaluating the simple stochastic integral:colorGreentextItô laa int_t_0^t W_sdd W_s raa approxsum_j laa W_j(W_j+1-W_j)raa\ncolorGreenhspace73em = sum_j laa W_j W_j+1raa - laa W_jW_jraa sim sum_j t_j - t_j = 0 com\ncolorMagentatextStratonovich laaint_t_0^t W_scircdd W_sraa approx sum_j laa frac12(W_j + W_j+1) (W_j+1-W_j)raa \ncolorMagentahspace73em = frac12sum_j laa W_j+1 W_j+1raa - laa W_j W_jraa  sim frac12sum_j t_j+1 - t_j = fract2perSDEs rarely can be solved in closed form; most often numerical solution of SDEs is brought to the rescue. Itô calculus has the advantage that is very easily implemented numerically. On the other hand, Stratonovich calculus coincides with that from normal calculus and this stems from the fact that it vies the white noise process as a series of colored noise processes with the de-correlation time tending to zero. This last fact is what made Stratonovich calculus more popular in the physics community. A nice discussion on the differences and similarities between the two calculi is done by van Kampen."
},

{
    "location": "forcing.html#A-simple-Stochastic-Differential-Equation-(SDE):-the-Ornstein–Uhlenbeck-process-1",
    "page": "Forcing",
    "title": "A simple Stochastic Differential Equation (SDE): the Ornstein–Uhlenbeck process",
    "category": "section",
    "text": "One of the simpler SDEs is the Ornstein–Uhlenbeck process. A variation of which is: \\begin{equation} 	x(t) = \\int{t0}^{t} -\\mu x(s)\\,\\dd s + \\int{t0}^{t} \\sqrt{\\sigma}\\,\\dd Ws \\per\\label{eq:OU} \\end{equation} Note that in differential form this is: \\begin{equation} 	\\dd xt = -\\mu xt \\,\\dd t + \\sqrt{\\sigma}\\,\\dd Ws \\per\\label{eq:1} \\end{equation} Luckily, here there is no need to distinguish between Itô and Stratonovich. This is because g is independent of x(t). But we stress that this is only a fortuitous However, this is often not the case.How do we time-step this SDE numerically? Let us assume a discretization of time into time-steps of tau: t_j=(j-1)tau. (What follows can be easily transfer to non-uniform time discretization.) With that, we denote x_jdefn x(t_j). Then the Euler–Mayorama time-step scheme for \\eqref{eq:1} is \\begin{equation} 	x{j+1} = xj + (-\\mu xj)\\tau + \\sqrt{\\sigma}(W{j+1}-W_j)\\per \\end{equation}Now let us ask the following question: How can we compute the work done by the noise? In other words, if we are interested in the evolution of the \"energy\" Edefn half x^2, how is the noise term attributing in the growth of E? To answer that we first have to find the SDE that energy E obeys. But, in doing so, it is important to adopt a single interpretation for computing stochastic integrals as now a transformation of variables is needed. That is, depending on whether we choose to interpret the stochastic integrals according to Itô or to Stratonovich calculus, E evolves as:\\begin{equation} {\\color{Green}\\text{Itô}:  \\dd Et  = \\left( -2\\mu Et + \\half \\sigma \\right)\\dd t  + xt \\sqrt{\\sigma}\\dd Wt}\\com\\label{eq:E_ito} \\end{equation}\\begin{equation} {\\color{Magenta}\\text{Stratonovich}: \\dd Et  = -2\\mu Et  \\dd t + xt\\circ \\sqrt{\\sigma}\\dd Wt}\\per\\label{eq:E_str} \\end{equation}How do we compute the work P done by the noise? Thus the work done by the stochastic forcing is:colorGreentextItô P_t = half sigma dd t + sqrtsigma x_t dd W_t approx  half sigma + sqrtsigma x_j (W_j+1-W_j)com\ncolorMagentatextStratonovich P_t =  x_t circsqrtsigma dd W_t approx sqrtsigma xleft(half(t_j+t_j+1)right)(W_j+1-W_j)perSay we didn\'t know the rules for transforming Stratonovich to Itô and we were wondering what is the extra drift term we have to include in the Itô formulations, i.e. the halfsigma term. We can compute the Itô\'s drift-term using that it is exactly equal to la x_tcircsqrtsigmadd W_tra; and for the latter we can use the \"usual\" calculus. That is, rewrite \\eqref{eq:OU} as:\\begin{equation} \\dot{x} = -\\mu x + \\xi\\com\\label{eq:OUcont} \\end{equation}where xi(t) is understood to be the \"continuous\" version of the white-noise process which is formally only understood in terms of distributions. The forcing xi has the properties:laa xi(t)raa = 0 quadtextandquad laa xi(t)xi(t)raa = sigma delta(t-t)perThus we need to compute la P_t ra = la x(t) xi(t) ra. But \\eqref{eq:OUcont} has formally the solution:x(t) = ee^-mu t x(0) + int_0^t ee^-mu(t-s)xi(s)dd sperand utilizing the above we getla P_t ra = la x(t) xi(t)  ra\n=  ee^-mu t underbracela x(0)xi(t)ra_=0 + int_0^t ee^-mu(t-s)la xi(t)xi(s)radd s\n= sigma int_0^t ee^-mu(t-s) delta(t-s)dd s =  fracsigma2 perAbove we used that int_0^tdelta(t-s)dd s = half, which is consistent with Stratonovich symmetric interpretation of stochastic integrals."
},

{
    "location": "forcing.html#Numerical-implementation-1",
    "page": "Forcing",
    "title": "Numerical implementation",
    "category": "section",
    "text": "How do we time-step \\eqref{eq:E_ito}? We use the Euler–Maruyama time-stepping scheme:	E_j+1 = E_j + left(-2mu E_j + fracsigma2right)tau + sqrtsigmax_j(W_j+1-W_j)perHowever, we cannot use Euler–Maruyama for time-stepping \\eqref{eq:Estr} since the Euler–Maruyama is \"Itô\"-thinking. To time-step \\eqref{eq:Estr} we have to approximate g in the middle of the time-step. There are many ways to do that, one of which is the, so called, Euler–Heun method:	widetildeE_j+1 = E_j + (-2mu E_j)tau + sqrtsigmax_j(W_j+1-W_j)com\n	E_j+1 = E_j + left(-2mu fracE_j+widetildeE_j+12right)tau + sqrtsigmafracx_j+x_j+12(W_j+1-W_j)per(Image: energy_comparison)Figure above shows a comparison of the energy evolution as done from:direct computation as half x_t^2,\ntime-integration of \\eqref{eq:E_ito}, and\ntime-integration of \\eqref{eq:E_str}.Figures below show the ensemble mean energy budgets (using 1000 ensemble members) as computed using Itô and Stratonovich. For the energy budget to close we have to be consistent: if we time-step the energy equation based on Stratonovich calculus then we must compute the work also according to Stratonovich. (For more details see examples/forcing/simpleSDEItoStratonovich.jl.(Image: energy_budgets_Ito) (Image: energy_budgets_Stratonovich)"
},

{
    "location": "forcing.html#A-simple-Stochastic-Partial-Differential-Equation-(SPDE)-1",
    "page": "Forcing",
    "title": "A simple Stochastic Partial Differential Equation (SPDE)",
    "category": "section",
    "text": "We want now to transfer all the knowledge we got from the previous sections to PDEs. In particular we\'ll focus on the simple SPDE:\\begin{equation} \\partial_t \\nabla^2\\psi(\\bx, t) =  -\\mu \\nabla^2\\psi(\\bx, t) + \\xi(\\bx,t)\\com\\label{eq:PDEcont} \\end{equation}which is also equivalently written as:\\begin{equation} \\dd \\nabla^2\\psit(\\bx) =  -\\mu \\nabla^2\\psit(\\bx) \\dd t + \\sqrt{\\sigma} \\dd W_t(\\bx) \\end{equation}The form \\eqref{eq:PDEcont} is the continuous version understood in the Stratonovich interpretation (similar to \\eqref{eq:OUcont}). Thus, forcing xi obeys now:laxi(bxt)ra = 0 quadtextandquadlaxi(bxt)xi(bxt) ra= Q(bx-bx)delta(t-t)comthat is the forcing is white in time but spatially correlated; its spatial correlation is prescribed by the function Q which is, necessarily, homogeneous in all its arguments (see discussion by Constantinou (see Appendix A).The above describes the vorticity evolution of a two-dimensional fluid nabla^2psi which is stochastically forced while dissipated by linear drag mu. The energy of the fluid is:E = halfoverlinegradpsi^2^xy = -halfoverlinepsinabla^2psi^xycomwhere the overbar denotes average over x and y. To obtain the energy equation we multiply \\eqref{eq:PDEcont} with -psi and average over the whole domain. Thus, the work done by the forcing is given by the term:P = -overlinepsixi^xycombut the above is a stochastic integral and it is meaningless without a rule for computing the stochastic integral.Numerically, the work done by the forcing can be obtained Stratonovich-wise as: \\begin{align} Pj = -\\,\\overline{\\frac{\\psi(\\bx,tj)+\\psi(\\bx,t{j+1})}{2}  \\xi(\\bx,t{j+1}) }^{x,y}\\com \\end{align} or Itô-wise \\begin{align} Pj = -\\,\\overline{ \\psi(\\bx,tj) \\xi(\\bx,t_{j+1}) }^{x,y} + \\text{drift}\\com \\end{align} But how much is the Itô drift term in this case? As in the previous section, the drift is precisely the ensemble mean of the Stratonovich work, i.e.:textrmIto drift= -overline launderbracepsi(bxt)circ  xi(bxt)_textrmStratonovich ra ^xycomBut again the above can be computed relatively easy if we use the \"formal\" solution of \\eqref{eq:PDEcont}:psi(bxt) = ee^-mu tpsi(bx0) + int_0^t ee^-mu(t-s)nabla^-2xi(bxs)dd scomwhich impliestextdrift = -overlineee^-mu tunderbracelaapsi(bx0) xi(bxt)raa_=0 ^xy - int_0^t ee^-mu(t-s)overlinenabla^-2laa xi(bxs)xi(bxt)raa^xydd s \n= -int_0^t ee^-mu(t-s)overlineunderbraceleftnabla^-2 Q(bx)rightbig_bx=0_textindependent of xydelta(t-s)^xydd s = -frac12 nabla^-2 Q(bx)big_bx=0 \n= -frac12 leftnabla^-2 int fracdd^2bk(2pi)^2 widehatQ(bk)ee^iibkbcdotbx right _bx=0\n= int fracdd^2bk(2pi)^2 fracwidehatQ(bk)2k^2perThus, the drift, or in this case the mean energy input rate by the stochastic forcing, is precisely determined from the spatial correlation of the forcing. Let us denote:varepsilon defn int fracdd^2bk(2pi)^2 fracwidehatQ(bk)2k^2perlabeleqdef_epsilonTherefore, work for a single forcing realization is computed numerically as:beginalign\ncolorGreentextItô colorGreen P_j  =  -overline psi(bxt_j) xi(bxt_j+1) ^xy  + varepsiloncom\ncolorMagentatextStratonovich  colorMagentaP_j = -overlinefracpsi(bxt_j)+psi(bxt_j+1)2  xi(bxt_j+1) ^xyper labeleqPtStrat\nendalignRemember, previously the work done by the stochastic forcing was:dd P_t = colorGreenfracsigma2dd t + sqrtsigmax_tdd W_t = colorMagentasqrtsigma x_tcircdd W_tcomand by sampling over various forcing realizations:langle dd P_trangle = fracsigma2dd t = langlesqrtsigma x_tcircdd W_trangleThe code uses Stratonovich. For example, the work done by the forcing in the TwoDTurb module is computed based on \\eqref{eq:PtStrat} with the function@inline function work(s, v::ForcedVars, g)\n  @. v.Uh = g.invKKrsq * (v.prevsol + s.sol)/2.0 * conj(v.Fh)\n  1/(g.Lx*g.Ly)*FourierFlows.parsevalsum(v.Uh, g)\nend"
},

{
    "location": "forcing.html#A-bit-less-simple-SPDE-1",
    "page": "Forcing",
    "title": "A bit-less-simple SPDE",
    "category": "section",
    "text": "It turns out that nothing changes if we include the nonlinear terms in the vorticity equation: \\begin{equation} \\partial_t \\nabla^2\\psi(\\bx, t) + \\J(\\psi,\\nabla^2\\psi) =  -\\mu \\nabla^2\\psi(\\bx, t) + \\xi(\\bx,t)\\per\\per\\label{eq:PDEcont2} \\end{equation} The nonlinearity does not alter the Itô drift; thus the ensemble mean energy input by the stochastic forcing, remains the same. We can easily verify this from the \"formal\" solution of \\eqref{eq:PDEcont2}:psi(bxt) = ee^-mu tpsi(bx0) + int_0^t ee^-mu(t-s)nabla^-2xi(bxs)dd s - int_0^t nabla^-2Jleft(psi(bxs)nabla^2psi(bxs)right)dd scomWhen multiplied with xi(bxt) the last term vanishes since its only non-zero contribution comes from the point s=t which is of measure zero (in the integrated sense).Figure below shows the energy budgets for a numerical solution of \\eqref{eq:PDEcont2}  starting from rest (psi(bx0)=0) in a doubly periodic square domain of size L (examples/twodturb/IsotropicRingForcing.jl). The forcing was prescribed to have power in a narrow ring in wavenumber space:widehatQ(bk)propto ee^-(bk-k_f)^2(2delta_f^2)comwith k_f L(2pi) = 12 and delta_f L(2pi) = 2. The mean energy input rate was set to varepsilon = 01.(Image: energy_budgets_SPDE_Stratonovich)"
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
    "text": "The equation is time-stepped forward in Fourier space:partial_t widehatq = - widehatJ(psi q) -left(mu k^2n_mu\n+nu k^2n_nuright) widehatq  + widehatf In doing so the Jacobian is computed in the conservative form: J(ab) = partial_y  (partial_x a) b -partial_x (partial_y a) b.Thus:mathcalL = -mu k^-2n_mu - nu k^2n_nu mathcalN(widehatq) = - mathrmik_x mathrmFFT(u q)-\n	mathrmik_y mathrmFFT(upsilon q) + widehatf "
},

{
    "location": "modules/twodturb.html#AbstractTypes-and-Functions-1",
    "page": "TwoDTurb Module",
    "title": "AbstractTypes and Functions",
    "category": "section",
    "text": "ParamsFor the unforced case (f=0) parameters AbstractType is build with Params and it includes:nu:   Float; viscosity or hyperviscosity coefficient.\nnnu: Integer0; the order of viscosity n_nu. Case n_nu=1 give normal viscosity.\nmu: Float; bottom drag or hypoviscosity coefficient.\nnmu: Integerge 0; the order of hypodrag n_mu. Case n_mu=0 give plain linear drag mu.For the forced case (fne 0) parameters AbstractType is build with ForcedParams. It includes all parameters in Params and additionally:calcF!: Function that calculates the forcing widehatfVarsFor the unforced case (f=0) variables AbstractType is build with Vars and it includes:q: Array of Floats; relative vorticity.\nU: Array of Floats; x-velocity, u.\nV: Array of Floats; y-velocity, v.\nsol: Array of Complex; the solution, widehatq.\nqh: Array of Complex; the Fourier transform widehatq.\nUh: Array of Complex; the Fourier transform widehatu.\nVh: Array of Complex; the Fourier transform widehatv.For the forced case (fne 0) variables AbstractType is build with ForcedVars. It includes all variables in Vars and additionally:Fh: Array of Complex; the Fourier transform widehatf.\nprevsol: Array of Complex; the values of the solution sol at the previous time-step (useful for calculating the work done by the forcing).calcN! functionThe nonlinear term mathcalN(widehatq) is computed via functions:calcN_advection!: computes - widehatJ(psi q) and stores it in array N.function calcN_advection!(N, sol, t, s, v, p, g)\n  @. v.Uh =  im * g.l  * g.invKKrsq * sol\n  @. v.Vh = -im * g.kr * g.invKKrsq * sol\n  @. v.qh = sol\n\n  A_mul_B!(v.U, g.irfftplan, v.Uh)\n  A_mul_B!s(v.V, g.irfftplan, v.Vh)\n  A_mul_B!(v.q, g.irfftplan, v.qh)\n\n  @. v.U *= v.q # U*q\n  @. v.V *= v.q # V*q\n\n  A_mul_B!(v.Uh, g.rfftplan, v.U) # \\hat{U*q}\n  A_mul_B!(v.Vh, g.rfftplan, v.V) # \\hat{U*q}\n\n  @. N = -im*g.kr*v.Uh - im*g.l*v.Vh\n  nothing\nendcalcN_forced!: computes - widehatJ(psi q) via calcN_advection! and then adds to it the forcing widehatf computed via calcF! function. Also saves the solution widehatq of the previous time-step in array prevsol.function calcN_forced!(N, sol, t, s, v, p, g)\n  calcN_advection!(N, sol, t, s, v, p, g)\n  if t == s.t # not a substep\n    v.prevsol .= s.sol # used to compute budgets when forcing is stochastic\n    p.calcF!(v.Fh, sol, t, s, v, p, g)\n  end\n  @. N += v.Fh\n  nothing\nendupdatevars!: uses sol to compute q, u, v, widehatu, and widehatv and stores them into corresponding arrays of Vars/ForcedVars."
},

{
    "location": "modules/twodturb.html#Examples-1",
    "page": "TwoDTurb Module",
    "title": "Examples",
    "category": "section",
    "text": "examples/twodturb/McWilliams.jl: A script that simulates decaying two-dimensional turbulence reproducing the results of the paper by\nMcWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. J. Fluid Mech., 146, 21-43.\nexamples/twodturb/IsotropicRingForcing.jl: A script that simulates stochastically forced two-dimensional turbulence. The forcing is temporally delta-corraleted and its spatial structure is isotropic with power in a narrow annulus of total radius k_f in wavenumber space."
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
    "location": "modules/barotropicqg.html#Examples-1",
    "page": "BarotropicQG Module",
    "title": "Examples",
    "category": "section",
    "text": "examples/barotropicqg/decayingbetaturb.jl: An script that simulates decaying quasi-geostrophic flow on a beta-plane demonstrating zonation.\nexamples/barotropicqg/forcedbetaturb.jl: An script that simulates forced-dissipative quasi-geostrophic flow on a beta-plane demonstrating zonation. The forcing is temporally delta-corraleted and its spatial structure is isotropic with power in a narrow annulus of total radius kf in wavenumber space.\nexamples/barotropicqg/ACConelayer.jl: A script that simulates barotropic quasi-geostrophic flow above topography reproducing the results of the paper by\nConstantinou, N. C. (2018). A barotropic model of eddy saturation. J. Phys. Oceanogr., 48 (2), 397-411."
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
    "text": "This module solves the advection diffusion equation for a passive tracer concentration c(x y t) in two-dimensions:partial_t c + boldsymbolu boldsymbolcdot boldsymbolnabla c = underbraceeta partial_x^2 c + kappa partial_y^2 c_textrmdiffusivity + underbracekappa_h (-1)^n_h nabla^2n_hc_textrmhyper-diffusivity where boldsymbolu = (uv) is the two-dimensional advecting flow, eta the x-diffusivity and kappa is the y-diffusivity. If eta is not defined then the code uses isotropic diffusivity, i.e., eta partial_x^2 c + kappa partial_y^2 cmapstokappanabla^2. The advecting flow could be either compressible or incompressible. "
},

{
    "location": "modules/traceradvdiff.html#Implementation-1",
    "page": "TracerAdvDiff Module",
    "title": "Implementation",
    "category": "section",
    "text": "The equation is time-stepped forward in Fourier space:partial_t widehatc = - widehatboldsymbolu boldsymbolcdot boldsymbolnabla c - left (eta k_x^2 + kappa k_y^2) +kappa_h k^2nu_h rightwidehatc Thus:beginalign*\nmathcalL = -eta k_x^2 - kappa k_y^2 - kappa_h k^2nu_h  \nmathcalN(widehatc) = - mathrmFFT(u partial_x c + upsilon partial_y c) \nendalign*"
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
    "location": "man/types.html#",
    "page": "Private types",
    "title": "Private types",
    "category": "page",
    "text": ""
},

{
    "location": "man/types.html#Private-types-1",
    "page": "Private types",
    "title": "Private types",
    "category": "section",
    "text": ""
},

{
    "location": "man/types.html#FourierFlows.Equation",
    "page": "Private types",
    "title": "FourierFlows.Equation",
    "category": "type",
    "text": "This type defines the linear implicit and explicit components of an equation. The linear implicit part of an equation is defined by an array of coefficients which multiply the solution. The explicit part of an equation is calculated by a function that may define linear and nonlinear parts.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#FourierFlows.Problem",
    "page": "Private types",
    "title": "FourierFlows.Problem",
    "category": "type",
    "text": "Problem(g, v, p, eq, ts)\n\nInitialize a FourierFlows problem on grid g, with variables v, parameters p, equation eq, and timestepper ts.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#FourierFlows.ZeroDGrid",
    "page": "Private types",
    "title": "FourierFlows.ZeroDGrid",
    "category": "type",
    "text": "ZeroDGrid()\n\nConstructs a placeholder grid object for \"0D\" problems (in other words, systems of ODEs).\n\n\n\n\n\n"
},

{
    "location": "man/types.html#Private-types-in-module-FourierFlows:-1",
    "page": "Private types",
    "title": "Private types in module FourierFlows:",
    "category": "section",
    "text": "Modules = [FourierFlows]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#FourierFlows.KuramotoSivashinsky.Vars-Tuple{Any}",
    "page": "Private types",
    "title": "FourierFlows.KuramotoSivashinsky.Vars",
    "category": "method",
    "text": "Returns the Vars object for Kuramoto-Sivashinsky.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#Private-types-in-module-KuramotoSivashinsky:-1",
    "page": "Private types",
    "title": "Private types in module KuramotoSivashinsky:",
    "category": "section",
    "text": "Modules = [FourierFlows.KuramotoSivashinsky]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#Private-types-in-module-TwoDTurb:-1",
    "page": "Private types",
    "title": "Private types in module TwoDTurb:",
    "category": "section",
    "text": "Modules = [FourierFlows.TwoDTurb]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#Private-types-in-module-BarotropicQG:-1",
    "page": "Private types",
    "title": "Private types in module BarotropicQG:",
    "category": "section",
    "text": "Modules = [FourierFlows.BarotropicQG]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#FourierFlows.TracerAdvDiff.ConstDiffParams",
    "page": "Private types",
    "title": "FourierFlows.TracerAdvDiff.ConstDiffParams",
    "category": "type",
    "text": "ConstDiffParams(eta, kap, kaph, nkaph, u, v)\nConstDiffParams(eta, kap, u, v)\n\nReturns the params for constant diffusivity problem with time-varying flow.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#FourierFlows.TracerAdvDiff.ConstDiffSteadyFlowParams",
    "page": "Private types",
    "title": "FourierFlows.TracerAdvDiff.ConstDiffSteadyFlowParams",
    "category": "type",
    "text": "ConstDiffSteadyFlowParams(eta, kap, kaph, nkaph, u, v, g)\nConstDiffSteadyFlowParams(eta, kap, u, v, g)\n\nReturns the params for constant diffusivity problem with time-steady flow.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#FourierFlows.TracerAdvDiff.Vars-Tuple{Any}",
    "page": "Private types",
    "title": "FourierFlows.TracerAdvDiff.Vars",
    "category": "method",
    "text": "Vars(g)\n\nReturns the vars for constant diffusivity problem on grid g.\n\n\n\n\n\n"
},

{
    "location": "man/types.html#Private-types-in-module-TracerAdvDiff:-1",
    "page": "Private types",
    "title": "Private types in module TracerAdvDiff:",
    "category": "section",
    "text": "Modules = [FourierFlows.TracerAdvDiff]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#Private-types-in-module-VerticallyFourierBoussinesq:-1",
    "page": "Private types",
    "title": "Private types in module VerticallyFourierBoussinesq:",
    "category": "section",
    "text": "Modules = [FourierFlows.VerticallyFourierBoussinesq]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/types.html#Private-types-in-module-VerticallyCosineBoussinesq:-1",
    "page": "Private types",
    "title": "Private types in module VerticallyCosineBoussinesq:",
    "category": "section",
    "text": "Modules = [FourierFlows.VerticallyCosineBoussinesq]\nPublic = false\nOrder = [:type]"
},

{
    "location": "man/functions.html#",
    "page": "Functions",
    "title": "Functions",
    "category": "page",
    "text": ""
},

{
    "location": "man/functions.html#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": ""
},

{
    "location": "man/functions.html#Base.resize!-Tuple{FourierFlows.AbstractDiagnostic,Int64}",
    "page": "Functions",
    "title": "Base.resize!",
    "category": "method",
    "text": "resize!(diag, newnum)\n\nResize the Diagnostic data and time arrays to length newnum.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.TimeStepper-Tuple{Any,Vararg{Any,N} where N}",
    "page": "Functions",
    "title": "FourierFlows.TimeStepper",
    "category": "method",
    "text": "Returns a time-stepper of type `steppernameTimeStepper\'.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.dealias!-Tuple{Any,OneDGrid}",
    "page": "Functions",
    "title": "FourierFlows.dealias!",
    "category": "method",
    "text": "dealias!(a, g, kalias)\n\nDealias a on the grid g with aliased x-wavenumbers kalias.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.gridpoints-Tuple{Any}",
    "page": "Functions",
    "title": "FourierFlows.gridpoints",
    "category": "method",
    "text": "gridpoints(g)\n\nReturns the collocation points of the grid g in 2D arrays X, Y.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.increment!-Tuple{FourierFlows.AbstractDiagnostic}",
    "page": "Functions",
    "title": "FourierFlows.increment!",
    "category": "method",
    "text": "increment!(diag)\nincrement!(diags)\n\nIncrement the Diagnostic diag, or an array of Diagnostics diags.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.savediagnostic-Tuple{FourierFlows.AbstractDiagnostic,String,String}",
    "page": "Functions",
    "title": "FourierFlows.savediagnostic",
    "category": "method",
    "text": "savediagnostic(diag, diagname)\n\nSave diagnostics to file, labeled by the string diagname.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.saveoutput-Tuple{Output}",
    "page": "Functions",
    "title": "FourierFlows.saveoutput",
    "category": "method",
    "text": "saveoutput(out)\n\nSave current output fields for file in out.filename.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.saveproblem-Tuple{AbstractProblem,String}",
    "page": "Functions",
    "title": "FourierFlows.saveproblem",
    "category": "method",
    "text": "saveproblem(prob, filename)\n\nSave certain aspects of a problem timestepper, grid, and params. Functions that are fields in params are not saved.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.stepforward!-Tuple{FourierFlows.Problem,AbstractArray,Any}",
    "page": "Functions",
    "title": "FourierFlows.stepforward!",
    "category": "method",
    "text": "stepforward!(prob, diags, nsteps)\n\nStep forward prob for nsteps, incrementing diagnostics in the array diags along the way.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.stepforward!-Tuple{FourierFlows.Problem,Any}",
    "page": "Functions",
    "title": "FourierFlows.stepforward!",
    "category": "method",
    "text": "stepforward!(prob, nsteps)\n\nStep forward prob for nsteps.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.stepforward!-Tuple{FourierFlows.Problem}",
    "page": "Functions",
    "title": "FourierFlows.stepforward!",
    "category": "method",
    "text": "stepforward!(prob)\n\nStep forward prob for one timestep.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.update!-Tuple{FourierFlows.AbstractDiagnostic}",
    "page": "Functions",
    "title": "FourierFlows.update!",
    "category": "method",
    "text": "update!(diag)\n\nUpdate diag with its current value.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#Functions-exported-from-FourierFlows:-1",
    "page": "Functions",
    "title": "Functions exported from FourierFlows:",
    "category": "section",
    "text": "Modules = [FourierFlows]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#FourierFlows.KuramotoSivashinsky.InitialValueProblem-Tuple{}",
    "page": "Functions",
    "title": "FourierFlows.KuramotoSivashinsky.InitialValueProblem",
    "category": "method",
    "text": "InitialValueProblem(; parameters...)\n\nConstruct an initial-value Kuramoto-Sivashinky problem that solves the equation\n\n∂t u + ∂ₓ⁴u + ∂ₓ²u + u ∂ₓu = 0.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.KuramotoSivashinsky.set_u!-NTuple{4,Any}",
    "page": "Functions",
    "title": "FourierFlows.KuramotoSivashinsky.set_u!",
    "category": "method",
    "text": "set_u!(prob, u)\nset_u!(s, v, g, u)\n\nSet the solution prob.state.sol as the transform of u and update variables.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.KuramotoSivashinsky.updatevars!-Tuple{Any,Any,Any}",
    "page": "Functions",
    "title": "FourierFlows.KuramotoSivashinsky.updatevars!",
    "category": "method",
    "text": "updatevars!(v, s, g)\n\nUpdate the vars in v on the grid g with the solution in s.sol.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#Functions-exported-from-KuramotoSivashinsky:-1",
    "page": "Functions",
    "title": "Functions exported from KuramotoSivashinsky:",
    "category": "section",
    "text": "Modules = [FourierFlows.KuramotoSivashinsky]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#Functions-exported-from-TwoDTurb:-1",
    "page": "Functions",
    "title": "Functions exported from TwoDTurb:",
    "category": "section",
    "text": "Modules = [FourierFlows.TwoDTurb]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#Functions-exported-from-BarotropicQG:-1",
    "page": "Functions",
    "title": "Functions exported from BarotropicQG:",
    "category": "section",
    "text": "Modules = [FourierFlows.BarotropicQG]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#FourierFlows.TracerAdvDiff.set_c!-NTuple{4,Any}",
    "page": "Functions",
    "title": "FourierFlows.TracerAdvDiff.set_c!",
    "category": "method",
    "text": "set_c!(s, v, g, c)\nset_c!(s, v, g, c::Function)\nset_c!(prob, c)\n\nSet the solution s.sol as the transform of c and update variables v on the grid g.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#FourierFlows.TracerAdvDiff.updatevars!-Tuple{Any,Any,Any}",
    "page": "Functions",
    "title": "FourierFlows.TracerAdvDiff.updatevars!",
    "category": "method",
    "text": "updatevars!(v, s, g)\n\nUpdate the vars in v on the grid g with the solution in s.sol.\n\n\n\n\n\n"
},

{
    "location": "man/functions.html#Functions-exported-from-TracerAdvDiff:-1",
    "page": "Functions",
    "title": "Functions exported from TracerAdvDiff:",
    "category": "section",
    "text": "Modules = [FourierFlows.TracerAdvDiff]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#Functions-exported-from-VerticallyFourierBoussinesq:-1",
    "page": "Functions",
    "title": "Functions exported from VerticallyFourierBoussinesq:",
    "category": "section",
    "text": "Modules = [FourierFlows.VerticallyFourierBoussinesq]\nPrivate = false\nOrder = [:function]"
},

{
    "location": "man/functions.html#Functions-exported-from-VerticallyCosineBoussinesq:-1",
    "page": "Functions",
    "title": "Functions exported from VerticallyCosineBoussinesq:",
    "category": "section",
    "text": "Modules = [FourierFlows.VerticallyCosineBoussinesq]\nPrivate = false\nOrder = [:function]"
},

]}
