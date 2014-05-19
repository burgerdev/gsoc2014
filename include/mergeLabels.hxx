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
          class LabelIterator, class LabelAccessor,
          class EqualityFunctor>
void mergeLabels(PixelIterator leftIter, SrcShape leftShape, PixelAccessor leftAcc,
                PixelIterator rightIter, SrcShape rightShape, PixelAccessor rightAcc,
                LabelIterator leftLabelsIter, SrcShape leftLabelsShape, LabelAccessor leftLabelsAcc,
                LabelIterator rightLabelsIter, SrcShape rightLabelsShape, LabelAccessor rightLabelsAcc,
                MultiArrayView<1, typename LabelAccessor::value_type> const & leftMap,
                MultiArrayView<1, typename LabelAccessor::value_type> const & rightMap,
                detail::UnionFindArray<typename LabelAccessor::value_type> & unionFind,
                EqualityFunctor equal)
{
    typedef typename LabelAccessor::value_type LabelType;

    //basically needed for iteration and border-checks
    int w = leftShape[0], h = leftShape[1], d = leftShape[2];
    int zLeft, zRight, zLabelsLeft, zLabelsRight;
    int yLeft, yRight, yLabelsLeft, yLabelsRight;
    int xLeft, xRight, xLabelsLeft, xLabelsRight;

    for(zLeft = 0, zRight = 0, zLabelsLeft = 0, zLabelsRight = 0;
        zLeft != d; // assume all shapes are the same
        ++zLeft, ++zRight, ++zLabelsLeft, ++zLabelsRight,
        ++leftIter.dim2(), ++rightIter.dim2(),
        ++leftLabelsIter.dim2(), ++rightLabelsIter.dim2())
    {
        PixelIterator leftItery(leftIter);
        PixelIterator rightItery(rightIter);
        LabelIterator leftLabelsItery(leftLabelsIter);
        LabelIterator rightLabelsItery(rightLabelsIter);
        
        for(yLeft = 0, yRight = 0, yLabelsLeft = 0, yLabelsRight = 0;
            yLeft != h; // assume all shapes are the same
            ++yLeft, ++yRight, ++yLabelsLeft, ++yLabelsRight,
            ++leftItery.dim1(), ++rightItery.dim1(),
            ++leftLabelsItery.dim1(), ++rightLabelsItery.dim1())
        {
            PixelIterator leftIterx(leftItery);
            PixelIterator rightIterx(rightItery);
            LabelIterator leftLabelsIterx(leftLabelsItery);
            LabelIterator rightLabelsIterx(rightLabelsItery);
            
            for(xLeft = 0, xRight = 0, xLabelsLeft = 0, xLabelsRight = 0;
                xLeft != w; // assume all shapes are the same
                ++xLeft, ++xRight, ++xLabelsLeft, ++xLabelsRight,
                ++leftIterx.dim0(), ++rightIterx.dim0(),
                ++leftLabelsIterx.dim0(), ++rightLabelsIterx.dim0())
            {
                if(equal(leftAcc(leftIterx), rightAcc(rightIterx)))
                {
                    if(leftLabelsAcc(leftLabelsIterx) > 0)
                    {
                        unionFind.makeUnion(leftMap[leftLabelsAcc(leftLabelsIterx)],
                                            rightMap[rightLabelsAcc(rightLabelsIterx)]);
                    }
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
    mergeLabels(left.first, left.second, left.third,
                right.first, right.second, right.third,
                leftLabels.first, leftLabels.second, leftLabels.third,
                rightLabels.first, rightLabels.second, rightLabels.third,
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

    mergeLabels(srcMultiArrayRange(left), 
                srcMultiArrayRange(right),
                srcMultiArrayRange(leftLabels), 
                srcMultiArrayRange(rightLabels),
                leftMap, rightMap, unionFind,
                std::equal_to<PixelType>());
}


template <class T> 
class UnionFunctor {
public:
    detail::UnionFindArray<T> uf_;
    const MultiArrayView<1, T> lm_;
    const MultiArrayView<1, T> rm_;
    
    UnionFunctor(detail::UnionFindArray<T> & unionFind,
                 const MultiArrayView<1, T> & lm, const MultiArrayView<1, T> & rm
                ) : uf_(unionFind), lm_(lm), rm_(rm) {}

    void operator()(const T & left, const T & right)
    {
        if (right > 0)
            uf_.makeUnion(lm_[left], rm_[right]);
    }
};

template <class T>
class CombiFunctor {
public:
    inline T  operator()(const T & left, const T & right, const T & label) const 
    {
        if (left == right)
            return label;
        else
            return 0;
    }
};
    

template <int N, class PixelType, class LabelType>
void mergeLabelsSimple(MultiArrayView<N, PixelType> const & left,
                 MultiArrayView<N, PixelType> const & right,
                 MultiArrayView<N, LabelType> const & leftLabels,
                 MultiArrayView<N, LabelType> const & rightLabels,
                 MultiArrayView<1, LabelType> const & leftMap,
                 MultiArrayView<1, LabelType> const & rightMap,
                 detail::UnionFindArray<LabelType> & unionFind
                 ) {
    vigra_precondition(leftLabels.shape() == rightLabels.shape(), "Shape mismatch");
    
    // functors for array combination
    UnionFunctor<LabelType> ufunc(unionFind, leftMap, rightMap);
    CombiFunctor<LabelType> cfunc;
    
    typedef typename MultiArrayView<N, PixelType>::difference_type TinyVec;
    
    // get the best order for first input (hope they are all the same)
    TinyVec strideOrder = left.strideOrdering();
    strideOrder = reverse(strideOrder);
    
    // order the arrays
    MultiArrayView<N, PixelType> leftOrdered = left.transpose(strideOrder);
    MultiArrayView<N, PixelType> rightOrdered = right.transpose(strideOrder);
    MultiArrayView<N, LabelType> leftLabelsOrdered = leftLabels.transpose(strideOrder);
    MultiArrayView<N, LabelType> rightLabelsOrdered = rightLabels.transpose(strideOrder);
    
    // bind at singleton axis  -- turns out this is not that efficient for my test data, but might be if I cut from larger arrays
    TinyVec shape = leftOrdered.shape();
//     int d = 0;
//     for (auto i = shape.begin(); i < shape.end(); ++i, ++d) 
//     {
//         if (*i == 1)
//         {
//             MultiArrayView<N-1, PixelType> leftOrderedBound = leftOrdered.bindAt(d, 0);
//             MultiArrayView<N-1, PixelType> rightOrderedBound = rightOrdered.bindAt(d, 0);
//             MultiArrayView<N-1, LabelType> leftLabelsOrderedBound = leftLabelsOrdered.bindAt(d, 0);
//             MultiArrayView<N-1, LabelType> rightLabelsOrderedBound = rightLabelsOrdered.bindAt(d, 0);
//             
//             MultiArray<N-1, LabelType> mask(leftOrderedBound.shape());
//             combineThreeMultiArrays(leftOrderedBound, rightOrderedBound, rightLabelsOrderedBound,  mask, cfunc);    
//             inspectTwoMultiArrays(leftLabelsOrderedBound, mask, ufunc);
//         }
//     }
    
    MultiArray<N, LabelType> mask(leftOrdered.shape());
    combineThreeMultiArrays(leftOrdered, rightOrdered, rightLabelsOrdered,  mask, cfunc);    
    inspectTwoMultiArrays(leftLabelsOrdered, mask, ufunc);
}

template <int N, class PixelType, class LabelType>
void mergeLabelsCoupled(MultiArrayView<N, PixelType> const & left,
                 MultiArrayView<N, PixelType> const & right,
                 MultiArrayView<N, LabelType> const & leftLabels,
                 MultiArrayView<N, LabelType> const & rightLabels,
                 MultiArrayView<1, LabelType> const & leftMap,
                 MultiArrayView<1, LabelType> const & rightMap,
                 detail::UnionFindArray<LabelType> & unionFind
                 ) {
    vigra_precondition(left.shape() == right.shape(), "Shape mismatch");
    vigra_precondition(leftLabels.shape() == rightLabels.shape(), "Shape mismatch");
    vigra_precondition(leftLabels.shape() == left.shape(), "Shape mismatch");
    
    const MultiArrayIndex LEFT_PIXEL = 1;
    const MultiArrayIndex RIGHT_PIXEL = 2;
    const MultiArrayIndex LEFT_LABEL = 3;
    const MultiArrayIndex RIGHT_LABEL = 4;
    
    LabelType left_label;
    LabelType right_label;
    
    
    typedef typename CoupledIteratorType<N, PixelType, PixelType, LabelType, LabelType>::type Iterator;
    
    Iterator start = createCoupledIterator(left, right, leftLabels, rightLabels);
    Iterator end = start.getEndIterator();
    
    //MAV.stride() == (1, 100, 10000)
    //std::cerr << left.stride() << std::endl;
    
    //bindAt
    // stride permutation in MAV2 = MAV.permuteStridesAscending()
    
    for (Iterator it = start; it < end; it++) 
    {
        if (it.get<LEFT_PIXEL>() == it.get<RIGHT_PIXEL>()) 
        {
            left_label = it.get<LEFT_LABEL>();
            right_label = it.get<RIGHT_LABEL>();
            if (left_label>0 && right_label>0) 
            {
                unionFind.makeUnion(leftMap[left_label], rightMap[right_label]);
            }
        }
    }
}

template <int N, class PixelType, class LabelType>
void mergeLabelsEvenWorse(MultiArrayView<N, PixelType> const & left,
                 MultiArrayView<N, PixelType> const & right,
                 MultiArrayView<N, LabelType> const & leftLabels,
                 MultiArrayView<N, LabelType> const & rightLabels,
                 MultiArrayView<1, LabelType> const & leftMap,
                 MultiArrayView<1, LabelType> const & rightMap,
                 detail::UnionFindArray<LabelType> & unionFind
                 ) {
    vigra_precondition(left.shape() == right.shape(), "Shape mismatch");
    vigra_precondition(leftLabels.shape() == rightLabels.shape(), "Shape mismatch");
    vigra_precondition(leftLabels.shape() == left.shape(), "Shape mismatch");
    
    const MultiArrayIndex LEFT_PIXEL = 1;
    const MultiArrayIndex RIGHT_PIXEL = 2;
    const MultiArrayIndex LEFT_LABEL = 3;
    const MultiArrayIndex RIGHT_LABEL = 4;
    
    typedef typename MultiArrayView<N, PixelType>::const_iterator PixelIterator;
    typedef typename MultiArrayView<N, LabelType>::const_iterator LabelIterator;
    
    PixelIterator end = left.end();
    PixelIterator left_it = left.begin();
    PixelIterator right_it = right.begin();
    LabelIterator llabels_it = leftLabels.begin();
    LabelIterator rlabels_it = rightLabels.begin();
    
    
    
    for (; left_it < end; left_it++, right_it++, llabels_it++, rlabels_it++) 
    {
        if (*left_it == *right_it) 
        {
            if (*llabels_it>0 && *rlabels_it>0) 
            {
                unionFind.makeUnion(leftMap[*llabels_it], rightMap[*rlabels_it]);
            }
        }
    }
}


} // namespace vigra



#endif
