#ifndef MERGELABELS_HXX
#define MERGELABELS_HXX

#include <vigra/tuple.hxx>

#include <vigra/multi_array.hxx>
#include <vigra/multi_shape.hxx>
#include <vigra/union_find.hxx>
#include <vigra/multi_iterator_coupled.hxx>

#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include <vigra/inspectimage.hxx>

#include <vigra/timing.hxx>

namespace vigra {

template <class PixelIterator, class LabelIterator, class LabelType,
          class Shape, class EqualityFunctor>
inline void
mergeLabels(PixelIterator left,
            PixelIterator right,
            LabelIterator leftLabels,
            LabelIterator rightLabels,
            const Shape & shape,
            MultiArrayView<1, LabelType> const & leftMap,
            MultiArrayView<1, LabelType> const & rightMap,
            detail::UnionFindArray<LabelType> & unionFind,
            EqualityFunctor equal, int n
           )
{
    int i = 0;
    if (n>0)
    {
        for(left.resetDim(n), right.resetDim(n), leftLabels.resetDim(n), rightLabels.resetDim(n);
            i < shape[n];
            left.incDim(n), right.incDim(n), leftLabels.incDim(n), rightLabels.incDim(n), i++)
        {
            mergeLabels(left, right, leftLabels, rightLabels, shape, leftMap, rightMap, unionFind, equal, n-1);
        }
    }
    else
    {
        LabelType * lmap = leftMap.data();
        LabelType * rmap = rightMap.data();
        for(left.resetDim(n), right.resetDim(n), leftLabels.resetDim(n), rightLabels.resetDim(n);
            i < shape[n];
            left.incDim(n), right.incDim(n), leftLabels.incDim(n), rightLabels.incDim(n), i++)
        {
            if (equal(*left, *right) && *leftLabels>0)
            {
                unionFind.makeUnion(lmap[*leftLabels], rmap[*rightLabels]);
            }
        }
    }
    left.resetDim(n), right.resetDim(n), leftLabels.resetDim(n), rightLabels.resetDim(n);
}

template <int N, class PixelType, class LabelType>
void
mergeLabels(MultiArrayView<N, PixelType> const & left,
            MultiArrayView<N, PixelType> const & right,
            MultiArrayView<N, LabelType> const & leftLabels,
            MultiArrayView<N, LabelType> const & rightLabels,
            MultiArrayView<1, LabelType> const & leftMap,
            MultiArrayView<1, LabelType> const & rightMap,
            detail::UnionFindArray<LabelType> & unionFind)
{
    vigra_precondition(left.shape() == right.shape(), "mergeLabels(): Data arrays shape mismatch");
    vigra_precondition(leftLabels.shape() == rightLabels.shape(), "mergeLabels(): Label arrays shape mismatch");
    vigra_precondition(leftLabels.shape() == left.shape(), "mergeLabels(): Labels/Data shape mismatch");
    vigra_precondition(leftMap.isUnstrided() && rightMap.isUnstrided(), "maps must be unstrided");

    typename MultiArrayView<N, PixelType>::difference_type strideOrder = left.strideOrdering();
    MultiArrayView<N, PixelType> leftReordered = left.transpose(strideOrder);
    mergeLabels(leftReordered.begin(), right.transpose(strideOrder).begin(),
                leftLabels.transpose(strideOrder).begin(), rightLabels.transpose(strideOrder).begin(),
                leftReordered.shape(),
                leftMap, rightMap, unionFind,
                std::equal_to<PixelType>(), N-1);
    
}


template <int N, class PixelType, class LabelType>
void
mergeLabelsRaw(MultiArrayView<N, PixelType> const & left,
            MultiArrayView<N, PixelType> const & right,
            MultiArrayView<N, LabelType> const & leftLabels,
            MultiArrayView<N, LabelType> const & rightLabels,
            MultiArrayView<1, LabelType> const & leftMap,
            MultiArrayView<1, LabelType> const & rightMap,
            detail::UnionFindArray<LabelType> & unionFind)
{
    vigra_precondition(left.shape() == right.shape(), "mergeLabels(): Data arrays shape mismatch");
    vigra_precondition(leftLabels.shape() == rightLabels.shape(), "mergeLabels(): Label arrays shape mismatch");
    vigra_precondition(leftLabels.shape() == left.shape(), "mergeLabels(): Labels/Data shape mismatch");
    vigra_precondition(leftMap.isUnstrided() && rightMap.isUnstrided(), "map arrays must be unstrided");
    vigra_precondition(left.isUnstrided() && right.isUnstrided(), "pixel arrays must be unstrided");
    vigra_precondition(leftLabels.isUnstrided() && rightLabels.isUnstrided(), "label arrays must be unstrided");
    
    
    PixelType* ldata = left.data();
    PixelType* rdata = right.data();
    
    LabelType* lldata = leftLabels.data();
    LabelType* rldata = rightLabels.data();
    
    LabelType* lmdata = leftMap.data();
    LabelType* rmdata = rightMap.data();
    
    typename MultiArrayView<N, PixelType>::difference_type_1 end = left.size();
    
    for (int i=0; i < end; i++)
    {
        if(ldata[i] == rdata[i])
        {
            if(lldata[i] > 0)
            {
                unionFind.makeUnion(lmdata[lldata[i]],
                                    rmdata[rldata[i]]);
            }
        }
    }
    
    
}




} // namespace vigra



#endif
