### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ c19e823c-5ee2-4c5e-882f-8611c6decb87
begin
	using PyCall
	using PyPlot
	PyPlot.svg(true)
	np = pyimport("numpy")
end

# ╔═╡ f2a3807f-271c-4049-bb20-46b410bdd5d2
using ForwardDiff

# ╔═╡ 4dbb0982-c4b8-4502-a3ee-02cba9d57cfa
using LinearAlgebra

# ╔═╡ 7cd43bd0-6a8a-4d7f-add1-53cebecd79de
begin
	filename = "pyactlike/pyactlike/data/bf_ACTPol_WMAP_lcdm.minimum.theory_cl"
	const tt_lmax = 5000
	const lmax_win = 7925
	cl = np.genfromtxt(
		filename,
		delimiter=nothing,
		unpack=false,
		max_rows=tt_lmax - 1,
		usecols=(0, 1, 2, 3))
	ell, dell_tt, dell_te, dell_ee = cl[:,1], cl[:,2], cl[:,3], cl[:,4]

	cltt, clte, clee = zeros(lmax_win), zeros(lmax_win), zeros(lmax_win)
	l_list = 2:tt_lmax
	invdlfac = (2π) ./ l_list ./ (l_list .+ 1)
	cltt[2:tt_lmax] .= dell_tt[begin:(tt_lmax - 1)] .* invdlfac
	clte[2:tt_lmax] .= dell_te[begin:(tt_lmax - 1)] .* invdlfac
	clee[2:tt_lmax] .= dell_ee[begin:(tt_lmax - 1)] .* invdlfac
end

# ╔═╡ c3de8359-835a-461e-8b3c-a6937721dbe4
size(cltt) # starts with ell = 1. index = ell

# ╔═╡ e4aeb22f-32b0-40fe-bc24-0c1125682aaa
struct ACTLike{T}
	win_d_tt::Matrix{T}
	win_d_te::Matrix{T}
	win_d_ee::Matrix{T}
	win_w_tt::Matrix{T}
	win_w_te::Matrix{T}
	win_w_ee::Matrix{T}
	fisher::Matrix{T}
	X_data::Vector{T}
end

# ╔═╡ f01786a8-5c7a-414b-9506-6002474e79cc
begin
	L = ACTLike{Float64}(
		np.load("pyactlike_data/win_func_d_tt.npy"),
		np.load("pyactlike_data/win_func_d_te.npy"),
		np.load("pyactlike_data/win_func_d_ee.npy"),
		np.load("pyactlike_data/win_func_w_tt.npy"),
		np.load("pyactlike_data/win_func_w_te.npy"),
		np.load("pyactlike_data/win_func_w_ee.npy"),
		np.load("pyactlike_data/fisher.npy"),
		np.load("pyactlike_data/X_data.npy")
	)
end;

# ╔═╡ 9af2d547-8493-40b1-916f-bee0f4f9604c
function like(L, cltt, clte, clee, yp2)
	
	lmax_win = 7925
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

# ╔═╡ 3ba7eff3-4f03-472b-a938-1a2dccd2a498
-2 * like(L, cltt, clte, clee, 1.003)

# ╔═╡ 036cc996-e068-4c41-ad26-af0a5a44e466
begin
	LCDM_pars = (
		# A_s=2.1824274E-09, 
		ln_1e10_A_s=log(1e10 * 2.1824274E-09),
		n_s=9.6437500E-01, theta_s_1e2=1.0419780, 
		omega_b=2.2283568E-02, tau_reio=7.3211982E-02, omega_dmeff=1.1988179E-01,
		# sigma_dmeff=0.0
	)
	offset_pars = (
		# A_s=2.1824274E-09 * 1.05, 
		ln_1e10_A_s=log(1e10 * 2.1824274E-09 * 1.05),
		n_s=9.6437500E-01 * 1.05, 
		theta_s_1e2=1.0419780 * 1.05,
		omega_b=2.2283568E-02 * 1.05, tau_reio=7.3211982E-02 * 1.05, omega_dmeff=1.1988179E-01 * 1.05, 
		# sigma_dmeff=3.0e-25
	)
	
	Δpar = NamedTuple{keys(LCDM_pars)}(
		map(k->(offset_pars[k]-LCDM_pars[k]), keys(LCDM_pars)))
end

# ╔═╡ d5977f2c-83dc-4308-997e-02a34dbb4069
T_CMB = 2.7255

# ╔═╡ f87f0e25-8fa9-41dd-8070-612721539068
begin
	pars = String.(keys(LCDM_pars))
	print(pars)
	p0 = Tuple(LCDM_pars)
	cl_keys = ("lcdm", pars...)
	
	cl_dict = Dict{String, Vector{Float64}}()

	for k in cl_keys
		cl0 = first(np.load("pyactlike_data/cl_$(k).npy", allow_pickle=true))
		for xy in ("tt", "te", "ee")
			cl_dict["$(k)_$(xy)"] = cl0.get(xy) * (T_CMB * 1e6)^2
		end
	end
end

# ╔═╡ 1fe7b28b-d044-479f-b9e4-f06da9a3bb8e
begin
	∂cl = Dict{String, Matrix{Float64}}()
	for xy in ("tt", "te", "ee")
	∂cl[xy] = hcat(
		[(cl_dict["$(p)_$(xy)"] .- cl_dict["lcdm_$(xy)"]) ./ Δpar[Symbol(p)] 
		for p in pars]...)
	end
end

# ╔═╡ 8d6e2777-5231-4669-9f99-4ba5b89fd54f
# begin
# 	clf()
# 	# plt.plot(cl_dict["A_s_tt"])
# 	plt.plot(∂cl["tt"][:,4] ./ cl_dict["lcdm_tt"])
# 	plt.yscale("symlog")
# 	gcf()
# end

# ╔═╡ 0be6a6e9-47d7-41e5-a188-f0eb879eb38b
function get_cl(p, xy)
	return cl_dict["lcdm_$(xy)"] .+ ∂cl[xy] * collect(p .- p0)
end

# ╔═╡ b2ad25c2-6223-4915-928a-b9097868a986
function par2like(p, L)
	cltt1, clte1, clee1 = zeros(lmax_win), zeros(lmax_win), zeros(lmax_win)

	spacer = zeros(lmax_win - 6000)
	cltt1 = [get_cl(p, "tt")[2:end]; spacer]
	clte1= [get_cl(p, "te")[2:end]; spacer]
	clee1 = [get_cl(p, "ee")[2:end]; spacer]

	return like(L, cltt1, clte1, clee1, 1.003)
end

# ╔═╡ e136ea2e-9932-4db8-804a-b9c3fd284e43
fish = -ForwardDiff.hessian(p->par2like(p,L), collect(p0))

# ╔═╡ 20c737f2-920c-46f0-8765-c5c2f132b958
fish[5,5] += 1 / (0.015)^2  # tau prior

# ╔═╡ 03e1d5a3-1b9e-4a4b-9e9f-2f2c931b3a2a
pars

# ╔═╡ c0d365d3-9059-4079-89d8-009a972b5a06
sqrt.(diag(inv(fish)))

# ╔═╡ c97e3e28-7f22-4f9c-9078-18d26e356133
p0

# ╔═╡ 76df51e6-5c45-4502-b3f3-411258b22c64


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

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

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
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

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
git-tree-sha1 = "c6cf981474e7094ce044168d329274d797843467"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.6"

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

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

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

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

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

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

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

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═c19e823c-5ee2-4c5e-882f-8611c6decb87
# ╠═7cd43bd0-6a8a-4d7f-add1-53cebecd79de
# ╠═c3de8359-835a-461e-8b3c-a6937721dbe4
# ╠═e4aeb22f-32b0-40fe-bc24-0c1125682aaa
# ╠═f01786a8-5c7a-414b-9506-6002474e79cc
# ╠═9af2d547-8493-40b1-916f-bee0f4f9604c
# ╠═3ba7eff3-4f03-472b-a938-1a2dccd2a498
# ╠═036cc996-e068-4c41-ad26-af0a5a44e466
# ╠═d5977f2c-83dc-4308-997e-02a34dbb4069
# ╠═f87f0e25-8fa9-41dd-8070-612721539068
# ╠═1fe7b28b-d044-479f-b9e4-f06da9a3bb8e
# ╠═8d6e2777-5231-4669-9f99-4ba5b89fd54f
# ╠═0be6a6e9-47d7-41e5-a188-f0eb879eb38b
# ╠═b2ad25c2-6223-4915-928a-b9097868a986
# ╠═f2a3807f-271c-4049-bb20-46b410bdd5d2
# ╠═e136ea2e-9932-4db8-804a-b9c3fd284e43
# ╠═20c737f2-920c-46f0-8765-c5c2f132b958
# ╠═4dbb0982-c4b8-4502-a3ee-02cba9d57cfa
# ╠═03e1d5a3-1b9e-4a4b-9e9f-2f2c931b3a2a
# ╠═c0d365d3-9059-4079-89d8-009a972b5a06
# ╠═c97e3e28-7f22-4f9c-9078-18d26e356133
# ╠═76df51e6-5c45-4502-b3f3-411258b22c64
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
