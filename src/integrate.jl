module Integrate

using Distributions
using FastGaussQuadrature

import Sobol: skip, SobolSeq
import Base.Iterators: take, Repeated
import HCubature: hcubatureyf
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


# using FastGaussQuadrature, LinearAlgebra
# import Base.Iterators: product, repeated
# function ∫q(f, dx::MvNormal; ndraw=100)
#   n = Int(ceil(ndraw^(1/length(dx))))
#   x, w = gausshermite(n)
#   L = cholesky(dx.Σ).L
#   sum(f(√2*L*vcat(xs...) + dx.μ)*prod(ws)
#       for (xs,ws) ∈ zip(product(repeated(x, length(dx))...),
#                         product(repeated(w, length(dx))...))
#         )/(π^(length(dx)/2))
# end

function QuadratureIntegrator(distribution::MvNormal, ndraw=100)
  n = Int(ceil(ndraw^(1/length(distribution))))
  x, w = gausshermite(n)
  L = cholesky(distribution.Σ).L
  sum(f(√2*L*vcat(xs...) + distribution.μ)*prod(ws) for (xs,ws) ∈ zip(product(repeated(x, length(distribution))...), 
                                                                      product(repeated(w, length(distribution))...))
        )/(π^(length(distribution)/2))

end

#print(QuadratureIntegrator(MvNormal([0.0,0.0],[1.0 0.5; 0.5 1.0]))(x->x[1]^2+x[2]^2))

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
    AdaptiveIntegrator(hcubature,x,Dx,args, limits)
end


end