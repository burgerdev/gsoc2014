// define PY_ARRAY_UNIQUE_SYMBOL (required by the numpy C-API)
#define PY_ARRAY_UNIQUE_SYMBOL lazycc_PyArray_API

#include <string>

// include the vigranumpy C++ API
#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

// include for the union find data structure
#include <vigra/union_find.hxx>

// my includes
#include "mergeLabels.hxx"

/* 
 * register converters for numpy scalars 
 * stolen from http://mathema.tician.de/software/pyublas/ (X11 license)
 * */

template <class T>
NPY_TYPES get_typenum();

template <>
NPY_TYPES get_typenum<npy_uint8>() { return NPY_UBYTE;}
template <>
NPY_TYPES get_typenum<npy_uint32>() { return NPY_UINT;}
template <>
NPY_TYPES get_typenum<npy_uint64>() { return NPY_ULONGLONG;}


template <class T>
const PyTypeObject *get_array_scalar_typeobj() {
    return (PyTypeObject *) PyArray_TypeObjectFromType(get_typenum<T>());
}

template <class T>
void *check_array_scalar(PyObject *obj) {
    if (obj->ob_type == get_array_scalar_typeobj<T>())
        return obj;
    else
        return 0;
}

template <class T>
static void convert_array_scalar(
        PyObject* obj,
        boost::python::converter::rvalue_from_python_stage1_data* data) {
    void* storage = ((boost::python::converter::rvalue_from_python_storage<T>*)data)->storage.bytes;

    // no constructor needed, only dealing with POD types
    PyArray_ScalarAsCtype(obj, reinterpret_cast<T *>(storage));

    // record successful construction
    data->convertible = storage;
}

template <class T>
void exportConverters() {
    // conversion of array scalars
    boost::python::converter::registry::push_back(
        check_array_scalar<T>
        , convert_array_scalar<T>
        , boost::python::type_id<T>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
        , get_array_scalar_typeobj<T>
#endif
        );
}


template <class T>
void exportVigraUnionFindArrayTyped(const char* name) {
    typedef vigra::detail::UnionFindArray<T> UnionFind;
    using namespace boost::python;
    
    exportConverters<T>();
    
    class_<UnionFind>(name, init<T>())
        .def("nextFreeLabel", &UnionFind::nextFreeLabel)
        .def("find", &UnionFind::find)
        .def("makeUnion", &UnionFind::makeUnion)
        .def("finalizeLabel", &UnionFind::finalizeLabel)
        .def("makeNewLabel", &UnionFind::makeNewLabel)
        .def("makeContiguous", &UnionFind::makeContiguous)
        .def("__getitem__", &UnionFind::operator[])
    ;
}

void exportVigraUnionFindArray() {
    exportVigraUnionFindArrayTyped<npy_uint8>("UnionFindUInt8");
    exportVigraUnionFindArrayTyped<npy_uint32>("UnionFindUInt32");
    exportVigraUnionFindArrayTyped<npy_uint64>("UnionFindUInt64");
}

namespace vigra {

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
     *   multidef("mergeLabels", pyMergeLabels2d<npy_uint8, npy_uint32, npy_uint64, float>(),
     *       (arg("left_image"),
     *       arg("right_image"),
     *       arg("left_labels"),
     *       arg("right_labels"),
     *       arg("left_mapping"),
     *       arg("right_mapping"),
     *       arg("UnionFind")),
     *       "Bla\n");
     */ 
}


// the argument of the init macro must be the module name
BOOST_PYTHON_MODULE_INIT(_lazycc_cxx)
{
    using namespace boost::python;

    // initialize numpy and vigranumpy
    vigra::import_vigranumpy();

    exportVigraUnionFindArray();
    
    exportMergeLabels();
}
