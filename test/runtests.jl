using Test

@testset "integrator" begin
    # include("../src/integrate.jl") # for interactive execution
    using Distributions

    dimx = 3
    A = rand(dimx,dimx)
    Σ = A*A'
    dx = MvNormal(zeros(dimx), Σ)

    # For hcubature, you need a high enough initdiv to get a good approximation. I changed 3 to 50. Else, the test fails.
    ∫a = Integrate.AdaptiveIntegrator(dx, options=(rtol=1e-6,initdiv=50))        
    V = ∫a(x->x*x')
    f(x) = exp(x[1])/sum(exp.(x))  
    @test isapprox(V, Σ, rtol=1e-4)
    
    val = ∫a(f)
    for N ∈ [1_000, 10_000, 100_000]
        ∫mc = Integrate.MonteCarloIntegrator(dx, N)
        ∫qmc = Integrate.QuasiMonteCarloIntegrator(dx,N)        
        ∫q = Integrate.QuadratureIntegrator(dx,N) 
        ∫sg = Integrate.SparseGridIntegrator(dx) 
        @test isapprox(∫mc(x->x*x'), Σ, rtol=10/sqrt(N))
        @test isapprox(∫qmc(x->x*x'), Σ, rtol=10/sqrt(N))
        @test isapprox(∫q(x->x*x'), Σ, rtol=10/sqrt(N))
        @test isapprox(∫sg(x->x*x'), Σ, rtol=10/sqrt(N))
      
        @test isapprox(∫mc(f),val,rtol=1/sqrt(N))
        @test isapprox(∫qmc(f),val,rtol=1/sqrt(N))
        @test isapprox(∫q(f),val,rtol=1/sqrt(N))
        @test isapprox(∫sg(f),val,rtol=1/sqrt(N))
    end
end

@testset "share=δ⁻¹" begin
    include("../src/blp.jl") 
    using LinearAlgebra
    J = 4
    dimx = 2
    dx = MvNormal(dimx, 1.0)
    Σ = [1 0.5; 0.5 1]
    N = 1_000
    ∫ = BLP.Integrate.QuasiMonteCarloIntegrator(dx, N)
    #∫ = Integrate.AdaptiveIntegrator(dx, options=(rtol=1e-6,initdiv=50))        
    X = [(-1.).^(1:J) 1:J]
    δ = collect((1:J)./J)
    s = BLP.share(δ,Σ,X,∫) 
    d = BLP.delta(s, Σ, X, ∫)
    @test d ≈ δ

    J = 10
    dimx = 4
    X = rand(J, dimx)
    dx = MvNormal(dimx, 1.0)
    Σ = I + ones(dimx,dimx)
    ∫ = BLP.Integrate.QuasiMonteCarloIntegrator(dx, N)
    #∫ = Integrate.AdaptiveIntegrator(dx, options=(rtol=1e-6,initdiv=50))        
    δ = 1*rand(J)
    s = BLP.share(δ,Σ,X,∫) 
    d = BLP.delta(s, Σ, X, ∫)
    @test isapprox(d, δ, rtol=1e-6)
    
    # Check that it works for small δ
    J = 10
    dimx = 4
    X = rand(J, dimx)
    dx = MvNormal(dimx, 1.0)
    Σ = I + ones(dimx,dimx)
    ∫ = BLP.Integrate.QuasiMonteCarloIntegrator(dx, N)
    #∫ = Integrate.AdaptiveIntegrator(dx, options=(rtol=1e-6,initdiv=50))        
    δ = 0.0001*rand(J)
    s = BLP.share(δ,Σ,X,∫) 
    d = BLP.delta(s, Σ, X, ∫)
    @test isapprox(d, δ, rtol=1e-6)

    # Check that it works for small s
    J = 10
    dimx = 4
    X = rand(J, dimx)
    dx = MvNormal(dimx, 1.0)
    Σ = I + ones(dimx,dimx)
    ∫ = BLP.Integrate.QuasiMonteCarloIntegrator(dx, N)
    #∫ = Integrate.AdaptiveIntegrator(dx, options=(rtol=1e-6,initdiv=50))        
    s = 0.1*rand(Dirichlet(1:(J+1)))[1:J]
    d = BLP.delta(s, Σ, X, ∫)
    s_hat = BLP.share(d, Σ, X, ∫)
    @test isapprox(s, s_hat, rtol=1e-6)

end
