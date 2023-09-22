using Distributions, ForwardDiff
numJ = 5
δ = rand(Uniform(0,1), numJ)
s = rand(Dirichlet(1:(numJ+1)))[1:numJ]
A = rand(2,2)
Σ = A*A'
#X = [(-1.).^(1:4) 1:4]
X = [(-2.).^(1:dimx) 1:dimx]
∫mc = Integrate.MonteCarloIntegrator(Distributions.MvNormal(zeros(size(Σ,1)), Σ), 1000)
∫q = Integrate.QuadratureIntegrator(Distributions.MvNormal(zeros(size(Σ,1)), Σ), 1000)
∫a = Integrate.AdaptiveIntegrator(Distributions.MvNormal(zeros(size(Σ,1)), Σ), options=(rtol=1e-6,initdiv=50))

s = share(δ, Σ, X, ∫mc)
s = share(δ, Σ, X, ∫q)
s = share(δ, Σ, X, ∫a)

delta(s, Σ, X, ∫mc)
delta(s, Σ, X, ∫sg)
delta(s, Σ, X, ∫a)
delta([0.5, 0.45, 0.0, 0.0, 0.0], Σ, X, ∫a)
delta([0.5, 0.45, 0.000001,0.0,0.0], Σ, X, ∫a)

include("../src/blp.jl") 
using LinearAlgebra
J = 10
dimx = 4
X = rand(J, dimx)
dx = MvNormal(dimx, 1.0)
Σ = I + ones(dimx,dimx)
#∫ = BLP.Integrate.QuasiMonteCarloIntegrator(dx, N)
∫ = Integrate.AdaptiveIntegrator(dx, options=(rtol=1e-6,initdiv=50))        
δ = 1*rand(J)
s = BLP.share(δ,Σ,X,∫) 
d = BLP.delta(s, Σ, X, ∫)
@test isapprox(d, δ, rtol=1e-6)