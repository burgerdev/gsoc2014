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

#include <iostream>

namespace vigra {

    
template <class PixelIterator, class PixelAccessor, class SrcShape,
          class LabelIterator, class LabelAccessor,class EqualityFunctor>
inline void
mergeLabels(PixelIterator leftIter, PixelAccessor leftAcc,
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
            mergeLabels<PixelIterator, PixelAccessor, NextSrcShape, 
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
        for (int i=0; i < width; 
        i++, leftIter.template dim<N-1>()++, rightIter.template dim<N-1>()++,
        leftLabelsIter.template dim<N-1>()++, rightLabelsIter.template dim<N-1>()++)
        {
            if(equal(leftAcc(leftIter), rightAcc(rightIter)))
            {
                if(leftLabelsAcc(leftLabelsIter) > 0)
                {
                    unionFind.makeUnion(leftMap[leftLabelsAcc(leftLabelsIter)],
                                        rightMap[rightLabelsAcc(rightLabelsIter)]);
                }
            }
            
        }
    }
}


template <class PixelIterator, class PixelAccessor, class SrcShape,
          class LabelIterator, class LabelAccessor,class EqualityFunctor>
inline void 
mergeLabels(triple<PixelIterator, SrcShape, PixelAccessor> left,
            triple<PixelIterator, SrcShape, PixelAccessor> right,
            triple<LabelIterator, SrcShape, LabelAccessor> leftLabels,
            triple<LabelIterator, SrcShape, LabelAccessor> rightLabels,
            MultiArrayView<1, typename LabelAccessor::value_type> const & leftMap,
            MultiArrayView<1, typename LabelAccessor::value_type> const & rightMap,
            detail::UnionFindArray<typename LabelAccessor::value_type> & unionFind, 
            EqualityFunctor equal)
{
    mergeLabels(left.first, left.third,
                right.first, right.third,
                leftLabels.first, leftLabels.third,
                rightLabels.first,rightLabels.third,
                left.second,
                leftMap, rightMap, unionFind, equal);
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
    
    for (int i=0; N > 1 && i < shape.size(); i++)
    {
        if (shape[i] == 1)
        {
            MultiArrayView<(N-1>1?N-1:1), PixelType> leftRed = left.bindAt(i, 0);
            MultiArrayView<(N-1>1?N-1:1), PixelType> rightRed = right.bindAt(i, 0);
            MultiArrayView<(N-1>1?N-1:1), LabelType> leftLabelsRed = leftLabels.bindAt(i, 0);
            MultiArrayView<(N-1>1?N-1:1), LabelType> rightLabelsRed = rightLabels.bindAt(i, 0);
            // sadly, this does not work because callign bindAt on a 1d array returns a 0d array (contrary to spec!)
//             mergeLabels<(N-1>1?N-1:1), PixelType, LabelType>(leftRed, rightRed, leftLabelsRed, rightLabelsRed,
//                         leftMap, rightMap, unionFind);
            mergeLabels(srcMultiArrayRange(leftRed), 
                        srcMultiArrayRange(rightRed),
                        srcMultiArrayRange(leftLabelsRed), 
                        srcMultiArrayRange(rightLabelsRed),
                        leftMap, rightMap, unionFind,
                        std::equal_to<PixelType>());
            return;
        }
    }
    
    mergeLabels(srcMultiArrayRange(left), 
                srcMultiArrayRange(right),
                srcMultiArrayRange(leftLabels), 
                srcMultiArrayRange(rightLabels),
                leftMap, rightMap, unionFind,
                std::equal_to<PixelType>());
    
//     mergeLabels(srcMultiArrayRange(leftOrdered), 
//                 srcMultiArrayRange(rightOrdered),
//                 srcMultiArrayRange(leftLabelsOrdered), 
//                 srcMultiArrayRange(rightLabelsOrdered),
//                 leftMap, rightMap, unionFind,
//                 std::equal_to<PixelType>());
}



} // namespace vigra



#endif
