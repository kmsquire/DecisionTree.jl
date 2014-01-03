module DecisionTree

using DataFrames
using DataStructures
using NumericExtensions

import Base: length, convert, promote_rule, show, start, next, done

export Leaf, Node, print_tree,
       build_stump, build_tree, prune_tree, apply_tree, nfoldCV_tree,
       build_forest, apply_forest, nfoldCV_forest,
       build_adaboost_stumps, apply_adaboost_stumps, nfoldCV_stumps,
       majority_vote, ConfusionMatrix, confusion_matrix

include("measures.jl")

immutable Leaf
    majority::Any
    probs::Dict
    values::Vector
end

immutable Node
    featid::Any
    featvals::Array
    children::Array{Union(Leaf,Node)}
end

convert(::Type{Node}, x::Leaf) = Node(0, Array[], [x])
promote_rule(::Type{Node}, ::Type{Leaf}) = Node
promote_rule(::Type{Leaf}, ::Type{Node}) = Node

function length(tree::Union(Leaf,Node))
    s = split(string(tree), "Leaf")
    return length(s) - 1
end

function print_tree(tree::Leaf, indent=0)
    matches = find(tree.values .== tree.majority)
    ratio = string(length(matches)) * "/" * string(length(tree.values))
    println("$(tree.majority) : $(ratio)")
end

function print_tree(tree::Node, indent=0)
    println("Feature $(tree.featid), Threshold $(tree.featval)")
    for (i,subtree) in enumerate(tree.children)
        print("    " ^ indent * "$i-> ")
        print_tree(subtree, indent + 1)
    end
end

const NO_BEST=(0,0)

function _split(labels::Vector, features::Matrix, nsubfeatures::Int, weights::Vector)
    if weights == [0]
        _split_info_gain(labels, features, nsubfeatures)
    else
        _split_neg_z1_loss(labels, features, nsubfeatures, weights)
    end
end

# Provide an iterator giving the unique values and corresponding ranges
# in a sorted vector
# Note: the vector is assumed to be sorted, and no checking is done!
immutable UniqueRanges
    v::Vector
    start::Int
    stop::Int
end

start(u::UniqueRanges) = u.start
done(u::UniqueRanges, s) = s > u.stop
next(u::UniqueRanges, s) = (val = u.v[s]; 
                            t = min(searchsortedlast(u.v, val, s, length(u.v), Base.Order.Forward), u.stop);
                            ((val, s:t), t+1))

UniqueRanges(v::Vector) = UniqueRanges(v, 1, length(v))

function fix_splits!(feature::Vector, splits::Vector)
    while length(splits) > 0 && searchsortedfirst(feature, splits[1]) == 1
        shift!(splits)
    end
    return unique(splits)
end

function _split_on{T}(labels::Vector, feature::Vector{T}, num_parts::Int=2)
    ord = sortperm(feature)
    feature = feature[ord]
    labels = labels[ord]
    
    uniq = unique(feature)

    N = length(uniq)
    part_size = N/num_parts
    initial_splits = uniq[iround([part_size+1:part_size:N])]
    fix_splits!(feature, initial_splits)

    splits = _split_sorted(labels, feature, initial_splits)

    return splits
end

function _split_sorted(labels::Vector, feature::Vector, splits::Vector, start::Int=1, stop::Int=length(labels))
    if length(splits) > 1
        splits_prev = copy(splits)
        iters = 1
        while true
            splice!(splits, 1:length(splits)-1, _split_sorted(labels, feature, splits[1:end-1], 1, searchsortedfirst(feature, splits[end])-1))
            splice!(splits, 2:length(splits), _split_sorted(labels, feature, splits[2:end], searchsortedfirst(feature, splits[1]), stop))
            fix_splits!(feature, splits)
            if splits == splits_prev || iters >= 100
                if iters >= 100
                    println("Max iters reached!")
                end
                break
            end
            iters += 1
            resize!(splits_prev, length(splits))
            copy!(splits_prev, splits)
        end
    elseif length(splits) == 1
        best = feature[1]
        best_val = -Inf

        hist1 = _hist(labels, 1:0)
        hist2 = _hist(labels, start:stop)
        N1 = 0
        N2 = stop-start+1

        for (d, range) in UniqueRanges(feature, start, stop)
            value = _info_gain(N1, hist1, N2, hist2)
            if value > best_val
                best_val = value
                best = d
            end

            deltaN = length(range)

            _hist_shift!(hist2, hist1, labels, range)
            N1 += deltaN
            N2 -= deltaN
        end

        splits[1] = best
    end

    return splits
end


function _split_info_gain(labels::Vector, features::Matrix, nsubfeatures::Int)
    nf = size(features, 2)
    N = length(labels)

    best = NO_BEST
    best_val = -Inf

    if nsubfeatures > 0
        inds = randperm(nf)[1:nsubfeatures]
    else
        inds = [1:nf]
    end

    for i in inds
        ord = sortperm(features[:,i])
        features_i = features[ord,i]
        labels_i = labels[ord]

        hist1 = _hist(labels_i, 1:0)
        hist2 = _hist(labels_i)
        N1 = 0
        N2 = N

        for (d, range) in UniqueRanges(features_i)
            value = _info_gain(N1, hist1, N2, hist2)
            if value > best_val
                best_val = value
                best = (i, d)
            end

            deltaN = length(range)

            _hist_shift!(hist2, hist1, labels_i, range)
            N1 += deltaN
            N2 -= deltaN
        end
    end
    return best
end

function _split_neg_z1_loss(labels::Vector, features::Matrix, nsubfeatures::Integer, weights::Vector)
    nf = size(features,2)
    best = NO_BEST
    best_val = -Inf
    if nsubfeatures > 0
        inds = randperm(nf)[1:nsubfeatures]
        nf = nsubfeatures
    else
        inds = [1:nf]
    end
    for i in 1:nf
        domain_i = sort(unique(features[:,inds[i]]))
        for d in domain_i[2:]
            cur_split = features[:,inds[i]] .< d
            value = _neg_z1_loss(labels[cur_split], weights[cur_split]) + _neg_z1_loss(labels[!cur_split], weights[!cur_split])
            if value > best_val
                best_val = value
                best = (inds[i], d)
            end
        end
    end
    return best
end

function build_stump(labels::Vector, features::Matrix, weights=[0])
    S = _split(labels, features, 0, weights)
    if S == NO_BEST
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    return Node(id, thresh,
                Leaf(majority_vote(labels[split]), labels[split]),
                Leaf(majority_vote(labels[!split]), labels[!split]))
end

function getprobs(labels::Vector)
    counts = counter(labels)
    total = sum(values(counts.map))
    return [key=>(value/total) for (key, value) in counts]
end

function build_tree(labels::Vector, data::DataFrame, feature_splits::Array{Tuple})
    if isempty(feature_splits)
        return Leaf(majority_vote(labels), getprobs(labels), labels)
    end

    (feature_name, parts) = feature_splits[1]
    feature = vec(array(data[:,feature_name]))
    if isa(parts, Array)
        featvals = parts
    else
        featvals = _split_on(labels, feature, parts)
    end
    if isempty(featvals)
        return Leaf(majority_vote(labels), getprobs(labels), labels)
    end

    idxs = split_idxs(labels, feature, featvals)
    child_label_sets = [labels[idx] for idx in idxs]
    child_data_sets = [data[idx, :] for idx in idxs]
   
    children = Union(Node, Leaf)[]

    for (child_labels, child_data) in zip(child_label_sets, child_data_sets)
        if length(child_labels) <= 50
            # probs calculated at current level, rather than child's level
            push!(children, Leaf(majority_vote(child_labels), getprobs(labels), child_labels))
        else
            push!(children, build_tree(child_labels, child_data, feature_splits[2:end]))
        end
    end

    Node(feature_name, featvals, children)
end

function build_tree(labels::Vector, features::Matrix, nsubfeatures=0)
    S = _split(labels, features, nsubfeatures, [0])
    if S == NO_BEST
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    labels_left = labels[split]
    labels_right = labels[!split]
    pure_left = all(labels_left .== labels_left[1])
    pure_right = all(labels_right .== labels_right[1])
    if pure_right && pure_left
        return Node(id, thresh,
                    Leaf(labels_left[1], labels_left),
                    Leaf(labels_right[1], labels_right))
    elseif pure_left
        return Node(id, thresh,
                    Leaf(labels_left[1], labels_left),
                    build_tree(labels_right,features[!split,:], nsubfeatures))
    elseif pure_right
        return Node(id, thresh,
                    build_tree(labels_left,features[split,:], nsubfeatures),
                    Leaf(labels_right[1], labels_right))
    else
        return Node(id, thresh,
                    build_tree(labels_left,features[split,:], nsubfeatures),
                    build_tree(labels_right,features[!split,:], nsubfeatures))
    end
end

function prune_tree(tree::Union(Leaf,Node), purity_thresh=1.0)
    function _prune_run(tree::Union(Leaf,Node), purity_thresh::Real)
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            all_labels = [tree.left.values, tree.right.values]
            majority = majority_vote(all_labels)
            matches = find(all_labels .== majority)
            purity = length(matches) / length(all_labels)
            if purity >= purity_thresh
                return Leaf(majority, all_labels)
            else
                return tree
            end
        else
            return Node(tree.featid, tree.featval,
                        _prune_run(tree.left, purity_thresh),
                        _prune_run(tree.right, purity_thresh))
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        pruned = _prune_run(tree, purity_thresh)
    end
    return pruned
end

function apply_tree(tree::Union(Leaf,Node), features::Vector)
    if typeof(tree) == Leaf
        return tree.majority
    elseif tree.featval == nothing
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::Leaf, features::SubDataFrame)
    return tree.probs[1]
end

function apply_tree(tree::Node, features::SubDataFrame)
    for (i,featval) in enumerate(tree.featvals)
        if features[tree.featid] < featval
            return apply_tree(tree.children[i], features)
        end
    end
    apply_tree(tree.children[end], features)
end

function apply_tree(tree::Union(Leaf,Node), features::Matrix)
    N = size(features,1)
    predictions = Array(Any,N)
    for i in 1:N
        predictions[i] = apply_tree(tree, squeeze(features[i,:],1))
    end
    return predictions
end

function apply_tree(tree::Union(Leaf,Node), features::DataFrame)
    N = size(features,1)
    predictions = Array(Float64,N)
    for (i,row) in enumerate(EachRow(features))
        predictions[i] = apply_tree(tree, row)
    end
    return predictions
end

function build_forest(labels::Vector, features::Matrix, nsubfeatures::Integer, ntrees::Integer, partialsampling=0.7)
    partialsampling = partialsampling > 1.0 ? 1.0 : partialsampling
    Nlabels = length(labels)
    Nsamples = int(partialsampling * Nlabels)
    forest = @parallel (vcat) for i in [1:ntrees]
        inds = rand(1:Nlabels, Nsamples)
        build_tree(labels[inds], features[inds,:], nsubfeatures)
    end
    return [forest]
end

function apply_forest{T<:Union(Leaf,Node)}(forest::Vector{T}, features::Vector)
    ntrees = length(forest)
    votes = Array(Any,ntrees)
    for i in 1:ntrees
        votes[i] = apply_tree(forest[i],features)
    end
    return majority_vote(votes)
end

function apply_forest{T<:Union(Leaf,Node)}(forest::Vector{T}, features::Matrix)
    N = size(features,1)
    predictions = Array(Any,N)
    for i in 1:N
        predictions[i] = apply_forest(forest, squeeze(features[i,:],1))
    end
    return predictions
end

function build_adaboost_stumps(labels::Vector, features::Matrix, niterations::Integer)
    N = length(labels)
    weights = ones(N) / N
    stumps = Node[]
    coeffs = FloatingPoint[]
    for i in 1:niterations
        new_stump = build_stump(labels, features, weights)
        predictions = apply_tree(new_stump, features)
        err = _weighted_error(labels, predictions, weights)
        new_coeff = 0.5 * log((1.0 + err) / (1.0 - err))
        matches = labels .== predictions
        weights[!matches] *= exp(new_coeff)
        weights[matches] *= exp(-new_coeff)
        weights /= sum(weights)
        push!(coeffs, new_coeff)
        push!(stumps, new_stump)
        if err < 1e-6
            break
        end
    end
    return (stumps, coeffs)
end

function apply_adaboost_stumps{T<:Union(Leaf,Node)}(stumps::Vector{T}, coeffs::Vector{FloatingPoint}, features::Vector)
    nstumps = length(stumps)
    counts = Dict()
    for i in 1:nstumps
        prediction = apply_tree(stumps[i], features)
        counts[prediction] = get(counts, prediction, 0.0) + coeffs[i]
    end
    top_prediction = None
    top_count = -Inf
    for i in collect(counts)
        if i[2] > top_count
            top_prediction = i[1]
            top_count = i[2]
        end
    end
    return top_prediction
end

function apply_adaboost_stumps{T<:Union(Leaf,Node)}(stumps::Vector{T}, coeffs::Vector{FloatingPoint}, features::Matrix)
    N = size(features,1)
    predictions = Array(Any,N)
    for i in 1:N
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, squeeze(features[i,:],1))
    end
    return predictions
end

end # module

