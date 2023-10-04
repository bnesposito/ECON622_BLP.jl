module Integrate

using Distributions, FastGaussQuadrature, LinearAlgebra, SparseGrids, StaticArrays, LoopVectorization

import Sobol: skip, SobolSeq
import Base.Iterators: take, Repeated
import HCubature: hcubature
import LinearAlgebra: cholesky
import Base.Iterators: product, repeated

abstract type AbstractIntegrator end

(∫::AbstractIntegrator)(f::Function) = sum(w*f(x) for (w,x) in zip(∫.w, ∫.x))

struct FixedNodeIntegrator{Tx,Tw} <: AbstractIntegrator
    x::Tx
    w::Tw
end

MonteCarloIntegrator(distribution::Distribution, ndraw=100)=FixedNodeIntegrator([rand(distribution) for i=1:ndraw], Repeated(1/ndraw))

function QuasiMonteCarloIntegrator(distribution::UnivariateDistribution, ndraws=100)
    ss = skip(SobolSeq(1), ndraw)
    x = [quantile(distribution, x) for x in take(ss,ndraw)]
    w = Repeated(1/ndraw)
    FixedNodeIntegrator(x,w)
end 

function QuasiMonteCarloIntegrator(distribution::AbstractMvNormal, ndraw=100)
    ss = skip(SobolSeq(length(distribution)), ndraw)
    L = cholesky(distribution.Σ).L
    x = [L*quantile.(Normal(), x) for x in take(ss,ndraw)]
    w = Repeated(1/ndraw)
    FixedNodeIntegrator(x,w)
end 

function QuadratureIntegrator(distribution::UnivariateDistribution, ndraw=100)
    n = Int(ceil(ndraw^(1/length(distribution))))
    x, w = gausshermite(n)
    w = w/π^(1/2)
    x = sqrt(2)*x
    FixedNodeIntegrator(x,w)
end

function QuadratureIntegrator(distribution::MvNormal, ndraw=100)
    n = Int(ceil(ndraw^(1/length(distribution))))
    x, w = gausshermite(n)
    L = cholesky(distribution.Σ).L
    wout = [prod(ws) for ws in product(repeated(w, length(distribution))...)]/π^(length(distribution)/2)
    xout = [sqrt(2)*L*vcat(xs...) + distribution.μ for xs in product(repeated(x, length(distribution))...)]
    FixedNodeIntegrator(xout, wout)
end

function QuadratureIntegrator_opt(distribution::MvNormal, ndraw=100)
    n = Int(ceil(ndraw^(1/length(distribution))))
    x, w = gausshermite(n)
    L = cholesky(distribution.Σ).L
    sizew = size(w,1)
    wout = @MMatrix zeros(eltype(w), sizew, sizew)
    #xout = Matrix{Vector{Float64}}(undef, sizew, sizew)

    @turbo for (cont, ws) in enumerate(product(repeated(w, length(distribution))...))
        wout[rem(cont-1, sizew) + 1, fld(cont-1, sizew) + 1] = prod(ws)
    end

    # No gain in performance.
    #@inbounds for (cont, xs) in enumerate(product(repeated(x, length(distribution))...))
    #    xout[rem(cont-1, sizew) + 1, fld(cont-1, sizew) + 1] = sqrt(2)*L*vcat(xs...) + distribution.μ
    #end

    @fastmath wout = wout./π^(length(distribution)/2)
    #wout = [prod(ws) for ws in product(repeated(w, length(distribution))...)]/π^(length(distribution)/2)
    xout = [sqrt(2)*L*vcat(xs...) + distribution.μ for xs in product(repeated(x, length(distribution))...)]
    FixedNodeIntegrator(xout, wout)
end

function SparseGridIntegrator(distribution::UnivariateDistribution, order=5)
    x, w = sparsegrid(length(distribution), order, gausshermite, sym=true)
    x = [sqrt(2)*a[1] for a in x]
    w = w/π^(1/2)
    FixedNodeIntegrator(x,w)
end

function SparseGridIntegrator(distribution::MvNormal, order = 5)
    x, w = sparsegrid(length(distribution), order, gausshermite, sym=true)
    L = cholesky(distribution.Σ).L
    wout = w/π^(length(distribution)/2)
    xout = [sqrt(2)*L*xs + distribution.μ for xs in x]
    FixedNodeIntegrator(xout,wout)
end

struct AdaptiveIntegrator{FE,FT,FJ,A,L} <: AbstractIntegrator
    eval::FE
    transform::FT
    detJ::FJ
    args::A
    limits::L
end

(∫::AdaptiveIntegrator)(f::Function) = ∫.eval(t->f(∫.transform(t))*∫.detJ(t), ∫.limits...; ∫.args...)[1]

function AdaptiveIntegrator(dist::AbstractMvNormal; eval=hcubature, options=())
    D = length(dist)
    x(t) = t./(1 .- t.^2)
    Dx(t) = prod((1 .+ t.^2)./(1 .- t.^2).^2)*pdf(dist,x(t))
    args = options
    limits = (-ones(D), ones(D))
    AdaptiveIntegrator(hcubature, x, Dx, args, limits)
end


end