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

    
template <class PixelIterator, class PixelAccessor, class SrcShape,
          class LabelIterator, class LabelAccessor,class EqualityFunctor>
inline void
mergeLabelsOldIter(PixelIterator leftIter, PixelAccessor leftAcc,
            PixelIterator rightIter, PixelAccessor rightAcc,
            LabelIterator leftLabelsIter, LabelAccessor leftLabelsAcc,
            LabelIterator rightLabelsIter, LabelAccessor rightLabelsAcc,
            SrcShape shape,
            MultiArrayView<1, typename LabelAccessor::value_type> const & leftMap,
            MultiArrayView<1, typename LabelAccessor::value_type> const & rightMap,
            detail::UnionFindArray<typename LabelAccessor::value_type> & unionFind,
            EqualityFunctor equal)
{
    static const int N = SrcShape::static_size;
    typedef typename SrcShape::value_type VTYPE;
    typedef typename LabelIterator::value_type LabelType;
    typedef TinyVector<VTYPE, (N-1>1 ? N-1 : 1)> NextSrcShape;
    
    
    int width = shape[N-1];
    
    if (N>1)
    {
        // call this function recursively on the reduced array
        for (int i=0; i < width; 
            i++, leftIter.template dim<N-1>()++, rightIter.template dim<N-1>()++,
            leftLabelsIter.template dim<N-1>()++, rightLabelsIter.template dim<N-1>()++)
        {
            PixelIterator leftIterNext(leftIter);
            PixelIterator rightIterNext(rightIter);
            LabelIterator leftLabelsIterNext(leftLabelsIter);
            LabelIterator rightLabelsIterNext(rightLabelsIter);
            
            NextSrcShape nextShape;
            nextShape.copy(shape);
            mergeLabelsOldIter<PixelIterator, PixelAccessor, NextSrcShape, 
                        LabelIterator, LabelAccessor, EqualityFunctor>(
                            leftIterNext, leftAcc,
                            rightIterNext, rightAcc,
                            leftLabelsIterNext, leftLabelsAcc,
                            rightLabelsIterNext, rightLabelsAcc, nextShape,
                            leftMap, rightMap, unionFind, equal);
                        
        }
    }
    else
    {
        LabelType* lm = leftMap.data();
        LabelType* rm = rightMap.data();
        for (int i=0; i < width; 
        i++, leftIter.template dim<N-1>()++, rightIter.template dim<N-1>()++,
        leftLabelsIter.template dim<N-1>()++, rightLabelsIter.template dim<N-1>()++)
        {
            if(equal(leftAcc(leftIter), rightAcc(rightIter)))
            {
                if(leftLabelsAcc(leftLabelsIter) > 0)
                {
                    unionFind.makeUnion(lm[leftLabelsAcc(leftLabelsIter)],
                                        rm[rightLabelsAcc(rightLabelsIter)]);
                }
            }
            
        }
    }
}


template <class PixelIterator, class PixelAccessor, class SrcShape,
          class LabelIterator, class LabelAccessor,class EqualityFunctor>
inline void 
mergeLabelsOldIter(triple<PixelIterator, SrcShape, PixelAccessor> left,
            triple<PixelIterator, SrcShape, PixelAccessor> right,
            triple<LabelIterator, SrcShape, LabelAccessor> leftLabels,
            triple<LabelIterator, SrcShape, LabelAccessor> rightLabels,
            MultiArrayView<1, typename LabelAccessor::value_type> const & leftMap,
            MultiArrayView<1, typename LabelAccessor::value_type> const & rightMap,
            detail::UnionFindArray<typename LabelAccessor::value_type> & unionFind, 
            EqualityFunctor equal)
{
    mergeLabelsOldIter(left.first, left.third,
                right.first, right.third,
                leftLabels.first, leftLabels.third,
                rightLabels.first,rightLabels.third,
                left.second,
                leftMap, rightMap, unionFind, equal);
}

template <int N, class PixelType, class LabelType>
inline void
mergeLabelsOldIter(MultiArrayView<N, PixelType> const & left,
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
    
    typedef typename MultiArrayView<N, PixelType>::difference_type TinyVec;
    
//     // get the best order for first input (hope they are all the same)
//     TinyVec strideOrder = left.strideOrdering();
//     //strideOrder = reverse(strideOrder);
//     
//     // order the arrays
//     std::cout << "Transposing to " << strideOrder << std::endl;
//     MultiArrayView<N, PixelType> leftOrdered = left.transpose(strideOrder);
//     MultiArrayView<N, PixelType> rightOrdered = right.transpose(strideOrder);
//     MultiArrayView<N, LabelType> leftLabelsOrdered = leftLabels.transpose(strideOrder);
//     MultiArrayView<N, LabelType> rightLabelsOrdered = rightLabels.transpose(strideOrder);

    typedef typename MultiArrayView<N, PixelType>::difference_type TinyVec;
    TinyVec shape = left.shape();
    
    
    mergeLabels(srcMultiArrayRange(left), 
                srcMultiArrayRange(right),
                srcMultiArrayRange(leftLabels), 
                srcMultiArrayRange(rightLabels),
                leftMap, rightMap, unionFind,
                std::equal_to<PixelType>());

}

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
            left.incDim(2), right.incDim(2), leftLabels.incDim(2), rightLabels.incDim(2), i++)
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
}

template <int N, class PixelType, class LabelType>
inline void
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

    
    mergeLabels(left.begin(), right.begin(),
                leftLabels.begin(), rightLabels.begin(),
                left.shape(),
                leftMap, rightMap, unionFind,
                std::equal_to<PixelType>(), N-1);
    
}


template <int N, class PixelType, class LabelType>
inline void
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
