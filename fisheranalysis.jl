### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 3707bc3c-e79d-11ec-2411-cdcf2d2bb7fc
begin
	using PyCall
	using PyPlot
	PyPlot.svg(true)
	np = pyimport("numpy")
end;

# ╔═╡ 5ee55741-2464-4a03-b357-c1fe00136dbf
using ForwardDiff, LinearAlgebra

# ╔═╡ 129714fd-46a2-4236-bf90-74b7ed331494
md"""
# Exact Fisher Analysis with Automatic Differentiation

This notebook tests the statistical power of ACT vs ACT+Planck, in the configurations reported by Aiola et al. 2020. I do this in a nice simple numerical way with automatic differentiation.

I directly implement two likelihoods below,
1. DR4 **pyactlike**, containing deep and wide patches
2. Heather Prince's MOPED compressed Planck 2018 likelihood. 

They're both Gaussian likelihoods, and I have tested that they match the originals to machine-precision. Next, I compute the **Hessian** of the likelihood numerically using forward-mode automatic differentiation. 

```math
F_{ij} = - \frac{ \partial^2 \mathcal{L}}{\partial \theta_i \, \partial \theta_j}
```

Armed with the Hessian obtained directly from the likelihoods, we can assess the statistical constraining power of ACT vs. ACT + Planck using a traditional Fisher analysis. I present some demonstrations that this test matches the parameter constraints provided in Aiola et al. 2020 to within ~20% (and for most parameters, significantly better!). 

"""

# ╔═╡ 87b65bd2-aec5-407a-b854-67418f48957c
md"""
## Likelihoods
You can skip the next few cells, which essentially just implement pyactlike twice.
"""

# ╔═╡ 2c4ef3fd-c8d0-451c-86f6-ee7cd7a40025
begin
	struct PlanckLike{T}
		blmin_TT::Vector{Int}
		blmax_TT::Vector{Int}
		bin_w_TT::Vector{T}
		blmin::Vector{Int}
		blmax::Vector{Int}
		bin_w::Vector{T}
		X_data::Vector{T}
		fisher::Matrix{T}
	end
	
	struct ACTLike{T}
		win_d_tt::Matrix{T}
		win_d_te::Matrix{T}
		win_d_ee::Matrix{T}
		win_w_tt::Matrix{T}
		win_w_te::Matrix{T}
		win_w_ee::Matrix{T}
		fisher::Matrix{T}
		X_data::Vector{T}
		lmax_win::Int
	end
	
	function like(L::ACTLike, cltt, clte, clee)
		yp2 = 1.003
		lmax_win = L.lmax_win
		cl_tt_d = L.win_d_tt * cltt[2:lmax_win]
		cl_te_d = L.win_d_te * clte[2:lmax_win]
		cl_ee_d = L.win_d_ee * clee[2:lmax_win]
		cl_tt_w = L.win_w_tt * cltt[2:lmax_win]
		cl_te_w = L.win_w_te * clte[2:lmax_win]
		cl_ee_w = L.win_w_ee * clee[2:lmax_win]
		
		b0 = 5
		nbintt = 40
		nbinte = 45
		nbinee = 45
		nbin = 260
	
		X_model = [
			cl_tt_d[b0+1 : b0 + nbintt]
			cl_te_d[begin:nbinte] * yp2
			cl_ee_d[begin:nbinee] * yp2 * yp2
			cl_tt_w[(b0+1) : (b0 + nbintt)]
			cl_te_w[begin:nbinte] * yp2
			cl_ee_w[begin:nbinee] * yp2 * yp2
		]
		
		Y = L.X_data .- X_model
		return -0.5 * (Y' * L.fisher * Y)
	end
	
	function like(L::PlanckLike, cltt, clte, clee)
		ellmin, nbintt, nbinte, nbinee = 2, 217, 199, 199
		plmin_TT, plmin = 2, 30
		calPlanck = 1
		Cltt_bin = [sum(cltt[
				(L.blmin_TT[i] .+ plmin_TT .- ellmin .+ 1):(L.blmax_TT[i] .+ plmin_TT .+ 1 .- ellmin)] .* L.bin_w_TT[(L.blmin_TT[i] + 1):L.blmax_TT[i]+1]) for i in 1:nbintt]
		Clte_bin = [sum(clte[
			(L.blmin[i] .+ plmin .- ellmin .+ 1):(L.blmax[i] .+ plmin .+ 1 .- ellmin)] .* L.bin_w[(L.blmin[i] + 1):L.blmax[i]+1]) for i in 1:nbinte]
		Clee_bin = [sum(clee[
			(L.blmin[i] .+ plmin .- ellmin .+ 1):(L.blmax[i] .+ plmin .+ 1 .- ellmin)] .* L.bin_w[(L.blmin[i] + 1):L.blmax[i]+1]) for i in 1:nbinee]
		X_model = [
			Cltt_bin ./ (calPlanck^2)
			Clte_bin ./ (calPlanck^2)
			Clee_bin ./ (calPlanck^2)
		]
		Y = L.X_data .- X_model
		return -0.5 * (Y' * L.fisher * Y)
	end
end

# ╔═╡ 47f8e2a2-1e09-4cbc-a3ac-6401608f8f6a
begin
	LA = ACTLike(
		np.load("pyactlike_data/win_func_d_tt.npy"),
		np.load("pyactlike_data/win_func_d_te.npy"),
		np.load("pyactlike_data/win_func_d_ee.npy"),
		np.load("pyactlike_data/win_func_w_tt.npy"),
		np.load("pyactlike_data/win_func_w_te.npy"),
		np.load("pyactlike_data/win_func_w_ee.npy"),
		np.load("pyactlike_data/fisher.npy"),
		np.load("pyactlike_data/X_data.npy"), 7925)
	
	LP = PlanckLike(
		np.load("pyactlike_data/planck_blmin_TT.npy"),
		np.load("pyactlike_data/planck_blmax_TT.npy"),
		np.load("pyactlike_data/planck_bin_w_TT.npy"),
		np.load("pyactlike_data/planck_blmin.npy"),
		np.load("pyactlike_data/planck_blmax.npy"),
		np.load("pyactlike_data/planck_bin_w.npy"),
		np.load("pyactlike_data/planck_X_data.npy"),
		np.load("pyactlike_data/planck_fisher.npy"))
end;

# ╔═╡ 4eb6707e-c6ce-4404-9d33-42bac16e53b0
md"""
## Model vector
The second derivative isn't really relevant here, but you do need a way to represent the derivative of your theory with respect to your parameters. I'm working hard on a way to get this directly (see Bolt.jl), but for now I just implement this via a first-order expansion of the theory code, by using finite differencing.

```math
C_\ell^{th} = C_{\ell}^{th}|_{p_0} + \frac{\Delta C_{\ell}^{th}}{\Delta \theta} (\theta - \theta_0).
```

Note that this analysis was entirely possible without automatic differentiation, but then I would have to write some complicated Fisher expressions with the binning operators.
"""

# ╔═╡ 8afd3e00-748f-426a-b884-4fc6a9ee35b0
begin
	const T_CMB = 2.7255
	LCDM_pars = (
		# A_s=2.1824274E-09, 
		ln_1e10_A_s=log(1e10 * 2.1824274E-09),
		n_s=9.6437500E-01, theta_s_1e2=1.0419780, 
		omega_b=2.2283568E-02, tau_reio=7.3211982E-02, omega_dmeff=1.1988179E-01,
		sigma_dmeff=0.0
	)
	offset_pars = (
		# A_s=2.1824274E-09 * 1.05, 
		ln_1e10_A_s=log(1e10 * 2.1824274E-09 * 1.05),
		n_s=9.6437500E-01 * 1.05, 
		theta_s_1e2=1.0419780 * 1.05,
		omega_b=2.2283568E-02 * 1.05, tau_reio=7.3211982E-02 * 1.05, omega_dmeff=1.1988179E-01 * 1.05, 
		sigma_dmeff=3.0e-25
	)
	Δpar = NamedTuple{keys(LCDM_pars)}(
		map(k->(offset_pars[k]-LCDM_pars[k]), keys(LCDM_pars)))
	pars = String.(keys(LCDM_pars))
	p0 = Tuple(LCDM_pars)
	cl_keys = ("lcdm", pars...)
	
	cl_dict = Dict{String, Vector{Float64}}()

	for k in cl_keys
		cl0 = first(np.load("pyactlike_data/cl_$(k).npy", allow_pickle=true))
		for xy in ("tt", "te", "ee")
			cl_dict["$(k)_$(xy)"] = cl0.get(xy) * (T_CMB * 1e6)^2
		end
	end
	∂cl = Dict{String, Matrix{Float64}}()
	for xy in ("tt", "te", "ee")
	∂cl[xy] = hcat(
		[(cl_dict["$(p)_$(xy)"] .- cl_dict["lcdm_$(xy)"]) ./ Δpar[Symbol(p)] 
		for p in pars]...)
	end
end

# ╔═╡ a9b86456-0a89-468d-bb4f-a1ad13107bdb
begin
	function get_cl(p, xy)
		return cl_dict["lcdm_$(xy)"] .+ ∂cl[xy] * collect(p .- p0)
	end
	
	function par2like(p, L::ACTLike, ttlmin=2)
		lmax_win = L.lmax_win
		
		spacer = zeros(lmax_win - 6000) # starts with ℓ=1
		cltt1 = [zeros(ttlmin - 2); get_cl(p, "tt")[ttlmin:end]; spacer] 
		clte1= [get_cl(p, "te")[2:end]; spacer]
		clee1 = [get_cl(p, "ee")[2:end]; spacer]
	
		return like(L, cltt1, clte1, clee1)
	end
	
	function par2like(p, L::PlanckLike)
		lmax_win = 2508
		cltt1 = get_cl(p, "tt")[3:end]   # starts with ℓ=2
		clte1= get_cl(p, "te")[3:end]
		clee1 = get_cl(p, "ee")[3:end]
	
		return like(L, cltt1, clte1, clee1)
	end
end

# ╔═╡ ecdc1673-a65a-4f5c-b820-c21794262468
md"""
## Automatic Differentiation (AD) and Fisher

Now, let's perform a Fisher analysis of these likelihoods. Here are the parameters.

"""

# ╔═╡ 0dde6b0e-fd79-4784-9403-f0a657e830b7
pars

# ╔═╡ aaadd7a7-6c78-4858-a90a-745efa778737
md"""
Now we will present the error bars on each parameter, with the last parameter being the error on the dark matter-baryon scattering cross section. In this case, we are using a 1 GeV, $n=0$ scattering model.
"""

# ╔═╡ 9f31c89e-3a24-43c5-aa38-6dea044d1ec1
md"""
#### Planck only:
"""

# ╔═╡ 3d51a515-084a-443e-913a-299495b39eed
begin
	fishP = -ForwardDiff.hessian(p->par2like(p, LP), collect(p0))
	fishP[5,5] += 1 / (0.015)^2  # tau prior to replace lowP
	sqrt.(diag(inv(fishP)))
end

# ╔═╡ 72abf005-1200-4afe-b654-84f2abfc3f45
md"""
#### ACT only:
"""

# ╔═╡ 14c66b90-7a4d-48c2-bf10-2e3b7fc144d2
begin
	fishA = -ForwardDiff.hessian(p->par2like(p, LA), collect(p0))
	fishA[5,5] += 1 / (0.015)^2  # tau prior to replace lowP
	sqrt.(diag(inv(fishA)))
end

# ╔═╡ 464034b6-b75f-4128-8cf8-3fd2d590a75f
md"""
#### ACT + Planck:
"""

# ╔═╡ 917cb116-4ff8-4707-b405-6ff12dccd54d
begin
	fishAP_A = -ForwardDiff.hessian(p->par2like(p, LA, 1800), collect(p0))
	fishAP_P = -ForwardDiff.hessian(p->par2like(p, LP), collect(p0))
	fishAP = fishAP_A .+ fishAP_P
	fishAP[5,5] += 1 / (0.015)^2  # tau prior to replace lowP
	sqrt.(diag(inv(fishAP)))
end

# ╔═╡ 48176533-1308-4e0c-8bb2-108c81edb996
md"""
Sadly, the last parameter is $\sigma_p$, the dark matter-baryon scattering cross section. It improves from $2.04 \times 10^{-25}$ cm$^2$ to $1.53 \times 10^{-25}$ cm$^2$, only a factor of 33%. That also implies that much of the improvement we see in the constraint is due to a statistical fluctuation between Planck and ACT.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"

[compat]
ForwardDiff = "~0.10.30"
PyCall = "~1.93.1"
PyPlot = "~2.10.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0-rc1"
manifest_format = "2.0"
project_hash = "61d1f052e6ce6500139ab07712fd3bb6738471c8"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "0f4e115f6f34bbe43c19751c90a38b2f380637b9"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.3"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.81.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "1fc929f47d7c151c839c5fc1375929766fb8edcc"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.1"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "14c1b795b9d764e1784713941e787e1384268103"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.10.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "383a578bdf6e6721f480e749d503ebc8405a0b22"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.6"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.41.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─129714fd-46a2-4236-bf90-74b7ed331494
# ╠═3707bc3c-e79d-11ec-2411-cdcf2d2bb7fc
# ╟─87b65bd2-aec5-407a-b854-67418f48957c
# ╠═2c4ef3fd-c8d0-451c-86f6-ee7cd7a40025
# ╠═47f8e2a2-1e09-4cbc-a3ac-6401608f8f6a
# ╟─4eb6707e-c6ce-4404-9d33-42bac16e53b0
# ╠═8afd3e00-748f-426a-b884-4fc6a9ee35b0
# ╠═a9b86456-0a89-468d-bb4f-a1ad13107bdb
# ╠═5ee55741-2464-4a03-b357-c1fe00136dbf
# ╟─ecdc1673-a65a-4f5c-b820-c21794262468
# ╠═0dde6b0e-fd79-4784-9403-f0a657e830b7
# ╟─aaadd7a7-6c78-4858-a90a-745efa778737
# ╟─9f31c89e-3a24-43c5-aa38-6dea044d1ec1
# ╠═3d51a515-084a-443e-913a-299495b39eed
# ╟─72abf005-1200-4afe-b654-84f2abfc3f45
# ╠═14c66b90-7a4d-48c2-bf10-2e3b7fc144d2
# ╟─464034b6-b75f-4128-8cf8-3fd2d590a75f
# ╠═917cb116-4ff8-4707-b405-6ff12dccd54d
# ╟─48176533-1308-4e0c-8bb2-108c81edb996
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
