module MNISTSims

import MLDatasets
using Random
using LinearAlgebra
using Statistics

macro spread(fn)
	:(args -> $fn(args...))
end

shuffledigits(digits) = digits[:, :, randperm(size(digits, 3))]

cosinesimilarity(a, b) = dot(a, b) / (norm(a) * norm(b))

function samplesimilarity(as, bs, fn)
	pairs = zip(eachslice(shuffledigits(as), dims=3),
		    eachslice(shuffledigits(bs), dims=3))
	fn(map(@spread(cosinesimilarity), pairs))
end

function similarities(samplesize)
	traindata = MLDatasets.MNIST(Tx=Float32)
	labels = traindata.targets
	digits = traindata.features

	digitgroups = map(eachcol(collect(0:9)' .== labels)) do mask
		all = digits[:, :, mask]
		all[:, :, randperm(samplesize)]
	end

	[samplesimilarity(la, lb, fn)
	  for la in digitgroups,
	      lb in digitgroups,
	      fn in (mean, minimum, maximum)]
end

end # module MNISTSims
