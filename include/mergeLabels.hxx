#ifndef MERGELABELS_HXX
#define MERGELABELS_HXX

// this must define the same symbol as the main module file (numpy requirement)
#define PY_ARRAY_UNIQUE_SYMBOL lazycc_PyArray_API
#define NO_IMPORT_ARRAY

#include <boost/python.hpp>

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


template <int N, class PixelType, class LabelType>
void mergeLabels(MultiArrayView<N, PixelType> const & left,
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
    
    typedef typename CoupledIteratorType<N, PixelType, PixelType, LabelType, LabelType>::type Iterator;
    
//     std::cerr << left.stride() << std::endl;
//     std::cerr << right.stride() << std::endl;
//     std::cerr << leftLabels.stride() << std::endl;
//     std::cerr << rightLabels.stride() << std::endl;
    
    Iterator start = createCoupledIterator(left, right, leftLabels, rightLabels);
    Iterator end = start.getEndIterator();
    
    for (Iterator it = start; it < end; it++) 
    {
        if (it.get<LEFT_PIXEL>() == it.get<RIGHT_PIXEL>()) 
        {
            unionFind.makeUnion(leftMap[it.get<LEFT_LABEL>()],
                                rightMap[it.get<RIGHT_LABEL>()]);
        }
    }
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




template <class PixelType>
inline void pythonMergeLabels3d(NumpyArray<3, Singleband<PixelType> > left,
                 NumpyArray<3, Singleband<PixelType> > right,
                 NumpyArray<3, Singleband<npy_uint32> > leftLabels,
                 NumpyArray<3, Singleband<npy_uint32> > rightLabels,
                 NumpyArray<1, Singleband<npy_uint32> > leftMap,
                 NumpyArray<1, Singleband<npy_uint32> > rightMap,
                 detail::UnionFindArray<npy_uint32> & unionFind) {
    
    //PyAllowThreads _pythread;
    mergeLabels<3, PixelType, npy_uint32>(left, right, leftLabels, rightLabels, leftMap, rightMap, unionFind);
}

VIGRA_PYTHON_MULTITYPE_FUNCTOR(pyMergeLabels3d, pythonMergeLabels3d)

template <class PixelType>
inline void pythonMergeLabels3dSimple(
                 NumpyArray<3, Singleband<PixelType> > left,
                 NumpyArray<3, Singleband<PixelType> > right,
                 NumpyArray<3, Singleband<npy_uint32> > leftLabels,
                 NumpyArray<3, Singleband<npy_uint32> > rightLabels,
                 NumpyArray<1, Singleband<npy_uint32> > leftMap,
                 NumpyArray<1, Singleband<npy_uint32> > rightMap,
                 detail::UnionFindArray<npy_uint32> & unionFind) {
    
    //PyAllowThreads _pythread;
    mergeLabelsSimple<3, PixelType, npy_uint32>(left, right, leftLabels, rightLabels, leftMap, rightMap, unionFind);
}

VIGRA_PYTHON_MULTITYPE_FUNCTOR(pyMergeLabels3dSimple, pythonMergeLabels3dSimple)


} // namespace vigra


void exportMergeLabels() {
    using namespace vigra;
    using namespace boost::python;
    multidef("mergeLabels", 
        pyMergeLabels3d<npy_uint8, npy_uint32, npy_uint64, float>(),
        (
            arg("left_image"), arg("right_image"),
            arg("left_labels"), arg("right_labels"),
            arg("left_mapping"), arg("right_mapping"),
            arg("UnionFind")
        ),
        "Bla\n");

    multidef("mergeLabelsSimple", 
        pyMergeLabels3dSimple<npy_uint8, npy_uint32, npy_uint64, float>(),
        (
            arg("left_labels"), arg("right_labels"),
            arg("left_mapping"), arg("right_mapping"),
            arg("UnionFind")
        ),
        "Bla\n");
    
    /*
    multidef("mergeLabels", pyMergeLabels2d<npy_uint8, npy_uint32, npy_uint64, float>(),
        (arg("left_image"),
        arg("right_image"),
        arg("left_labels"),
        arg("right_labels"),
        arg("left_mapping"),
        arg("right_mapping"),
        arg("UnionFind")),
        "Bla\n");
    */ 
}

#endif
