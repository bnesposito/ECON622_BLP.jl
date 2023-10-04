using Distributions, CodeTracking, BenchmarkTools, Profile

include("../src/blp.jl") 
include("../src/integrate.jl")

δ = [1.0, 2.0, 3.0]
Σ = [1.0 0.5; 0.5 1.0]
x = [1.0 0.0; 0.0 1.0; 3.0 2.0]
∫ = Integrate.QuadratureIntegrator(MvNormal(2,1.0), 1_000)
@code_string BLP.share(δ, Σ, x, ∫)

@btime BLP.share(δ, Σ, x, ∫)

Profile.clear()
Profile.init(n=10^7, delay = 0.000001)
@profile out = BLP.share(δ, Σ, x, ∫)
Profile.print(noisefloor = 1.0) 
@profview BLP.share(δ, Σ, x, ∫)
# It seems like most of the time is spent on the shareν function.

@btime ∫ = Integrate.QuadratureIntegrator(MvNormal(2,1.0), 1000)
@btime ∫opt = Integrate.QuadratureIntegrator_opt(MvNormal(2,1.0), 1000)

∫ = Integrate.QuadratureIntegrator(MvNormal(2,1.0), 1000)
∫opt = Integrate.QuadratureIntegrator_opt(MvNormal(2,1.0), 1000)

@code_warntype BLP.share(δ, Σ, x, ∫)
@code_warntype BLP.share(δ, Σ, x, ∫opt)

@btime BLP.share(δ, Σ, x, ∫)
@btime BLP.share_opt(δ, Σ, x, ∫)
@btime BLP.share_opt(δ, Σ, x, ∫opt)

# Same results but no improvement in performance unfortunately. Even worse, the optimized version is slower. 