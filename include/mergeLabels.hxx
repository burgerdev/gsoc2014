#ifndef MERGELABELS_HXX
#define MERGELABELS_HXX

// this must define the same symbol as the main module file (numpy requirement)
#define PY_ARRAY_UNIQUE_SYMBOL lazycc_PyArray_API
#define NO_IMPORT_ARRAY

#include <boost/python.hpp>

#include <vigra/multi_array.hxx>
#include <vigra/union_find.hxx>
#include <vigra/multi_iterator_coupled.hxx> 

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
    
    static const MultiArrayIndex l = 1;
    static const MultiArrayIndex r = 2;
    static const MultiArrayIndex ll = 3;
    static const MultiArrayIndex rl = 4;
    
    
    typedef typename CoupledIteratorType<N, PixelType, PixelType, LabelType, LabelType>::type Iterator;
    
    Iterator start = createCoupledIterator(left, right, leftLabels, rightLabels);
    Iterator end = start.getEndIterator();
    
    for (Iterator it = start; it < end; it++) {
        //std::cout << "Inspecting index " << it.get<0>() << std::endl;
        if (it.get<l>() == it.get<r>() && it.get<ll>() > 0 && it.get<rl>() > 0) {
            unionFind.makeUnion(leftMap[it.get<ll>()], rightMap[it.get<rl>()]);
            
            //std::cout << "merging labels " << leftMap[it.get<ll>()] << " and " << rightMap[it.get<rl>()] << std::endl;
        }
    }
}

} // namespace vigra



template <int N, class P, class L>
void exportLabelType() {
    using namespace boost::python;
    def("mergeLabels",
        vigra::registerConverters(&vigra::mergeLabels<N, P, L>),
        (arg("X"), 
         arg("Y"),
         arg("labelsX"), 
         arg("labelsY"),
         arg("mapX"), 
         arg("mapY"),
         arg("unionFind")),
        "TODO\n");
}

template <int N, class P>
void exportPixelType() {
    exportLabelType<N, P, vigra::UInt8>();
    exportLabelType<N, P, vigra::UInt32>();
    exportLabelType<N, P, vigra::UInt64>();
}

template <int N>
void exportDim() {
    exportPixelType<N, vigra::UInt8>();
    exportPixelType<N, vigra::UInt32>();
    exportPixelType<N, vigra::UInt64>();
}

void exportMergeLabels() {
    exportDim<1>();
    exportDim<2>();
    exportDim<3>();
    exportDim<4>();
    exportDim<5>();
}

#endif
