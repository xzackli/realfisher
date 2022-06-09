### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 1bd0650e-e793-11ec-2af7-a10256644fca
begin
	using PyCall
	using PyPlot
	PyPlot.svg(true)
	np = pyimport("numpy")
end

# ╔═╡ eb07a340-16a1-4be3-beaf-4f772f624e29
begin
	filename = "/home/zack/src/planck-lite-py/data/Dl_planck2015fit.dat"
	cls = np.genfromtxt(filename)
	ls, Dltt, Dlte, Dlee = cls[:,1], cls[:,2], cls[:,3], cls[:,4]
	# starts with l=2
	fac = ls .* (ls .+ 1) ./ (2π)
	Cltt=Dltt ./ fac
	Clte=Dlte ./ fac
	Clee=Dlee ./ fac
end

# ╔═╡ 0260b736-5bbf-4d1d-b931-2c817f70c08f
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

# ╔═╡ ca27f086-8d60-4d33-bbef-ad82e706ecaf
begin
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

# ╔═╡ 0d5d4bb6-d440-428e-8693-c838a16753fd
function plancklike(L, cltt, clte, clee)
	ellmin, nbintt, nbinte, nbinee = 2, 217, 199, 199
	plmin_TT, plmin = 2, 30
	calPlanck = 1
	Cltt_bin = [sum(Cltt[
			(LP.blmin_TT[i] .+ plmin_TT .- ellmin .+ 1):(LP.blmax_TT[i] .+ plmin_TT .+ 1 .- ellmin)] .* LP.bin_w_TT[(LP.blmin_TT[i] + 1):LP.blmax_TT[i]+1]) for i in 1:nbintt]
	Clte_bin = [sum(Clte[
		(LP.blmin[i] .+ plmin .- ellmin .+ 1):(LP.blmax[i] .+ plmin .+ 1 .- ellmin)] .* LP.bin_w[(LP.blmin[i] + 1):LP.blmax[i]+1]) for i in 1:nbinte]
	Clee_bin = [sum(Clee[
		(LP.blmin[i] .+ plmin .- ellmin .+ 1):(LP.blmax[i] .+ plmin .+ 1 .- ellmin)] .* LP.bin_w[(LP.blmin[i] + 1):LP.blmax[i]+1]) for i in 1:nbinee]
	X_model = [
		Cltt_bin ./ (calPlanck^2)
		Clte_bin ./ (calPlanck^2)
		Clee_bin ./ (calPlanck^2)
	]
	Y = LP.X_data .- X_model
	return -0.5 * (Y' * LP.fisher * Y)
end

# ╔═╡ 451b9514-4df9-4671-ba28-7f0b033f1057
plancklike(LP, Cltt, Clte, Clee) - (-293.9558650179513)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"

[compat]
PyCall = "~1.93.1"
PyPlot = "~2.10.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0-rc1"
manifest_format = "2.0"
project_hash = "e10c2652d96de7a92f1185cb161cc660c8fa3ef3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

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

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

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

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

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

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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
"""

# ╔═╡ Cell order:
# ╠═1bd0650e-e793-11ec-2af7-a10256644fca
# ╠═eb07a340-16a1-4be3-beaf-4f772f624e29
# ╠═0260b736-5bbf-4d1d-b931-2c817f70c08f
# ╠═ca27f086-8d60-4d33-bbef-ad82e706ecaf
# ╠═0d5d4bb6-d440-428e-8693-c838a16753fd
# ╠═451b9514-4df9-4671-ba28-7f0b033f1057
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
