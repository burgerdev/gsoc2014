
#include <vigra/multi_array.hxx>
#include <vigra/metaprogramming.hxx>
#include <vigra/inspector_passes.hxx>


namespace vigra {

template <class Iterator1, class Accessor1, 
          class Iterator2, class Accessor2,
          class Iterator3, class Accessor3, 
          class Iterator4, class Accessor4, 
          class Functor>
void
inspectFourLines(Iterator1 s1,
                Iterator1 s1end, Accessor1 src1,
                Iterator2 s2, Accessor2 src2,
                Iterator3 s3, Accessor3 src3,
                Iterator4 s4, Accessor4 src4,
                Functor & f)
{
    for(; s1 != s1end; ++s1, ++s2, ++s3, ++s4)
        f(src1(s1), src2(s2), src3(s3), src4(s4));
}


template <class Iterator1, class Shape, class Accessor1, 
          class Iterator2, class Accessor2,
          class Iterator3, class Accessor3, 
          class Iterator4, class Accessor4, 
          class Functor>
inline void
inspectFourMultiArraysImpl(Iterator1 s1, Shape const & shape, Accessor1 a1,
                          Iterator2 s2, Accessor2 a2,
                          Iterator3 s3, Accessor3 a3,
                          Iterator4 s4, Accessor4 a4,
                          Functor & f, MetaInt<0>)
{
    inspectFourLines(s1, s1 + shape[0], a1, s2, a2, s3, a3, s4, a4, f);
}
    
template <class Iterator1, class Shape, class Accessor1, 
          class Iterator2, class Accessor2, 
          class Iterator3, class Accessor3, 
          class Iterator4, class Accessor4,
          class Functor, int N>
void
inspectFourMultiArraysImpl(Iterator1 s1, Shape const & shape, Accessor1 a1,
                          Iterator2 s2, Accessor2 a2,
                          Iterator3 s3, Accessor3 a3,
                          Iterator4 s4, Accessor4 a4,
                          Functor & f, MetaInt<N>)
{
    Iterator1 s1end = s1 + shape[N];
    for(; s1 < s1end; ++s1, ++s2, ++s3, ++s4)
    {
        inspectFourMultiArraysImpl(s1.begin(), shape, a1, 
                                  s2.begin(), a2, 
                                  s3.begin(), a3, 
                                  s4.begin(), a4,
                                  f, MetaInt<N-1>());
    }
}


template <class Iterator1, class Shape, class Accessor1,
          class Iterator2, class Accessor2,
          class Iterator3, class Accessor3, 
          class Iterator4, class Accessor4>
struct inspectFourMultiArrays_binder
{
    Iterator1     s1;
    const Shape & shape;
    Accessor1     a1;
    Iterator2     s2;
    Accessor2     a2;
    Iterator3     s3;
    Accessor3     a3;
    Iterator4     s4;
    Accessor4     a4;
    inspectFourMultiArrays_binder(Iterator1 s1_, const Shape & shape_,
                                  Accessor1 a1_, Iterator2 s2_, Accessor2 a2_,
                                  Iterator3 s3_, Accessor3 a3_,
                                  Iterator4 s4_, Accessor4 a4_)
        : s1(s1_), shape(shape_), a1(a1_), s2(s2_), a2(a2_), s3(s3_), a3(a3_) , s4(s4_), a4(a4_) {}
    template <class Functor>
    void operator()(Functor & f)
    {
        inspectFourMultiArraysImpl(s1, shape, a1, s2, a2, s3, a3, s4, a4, f,
                                  MetaInt<Iterator1::level>());
    }
};
    
template <class Iterator1, class Shape, class Accessor1,
          class Iterator2, class Accessor2,
          class Iterator3, class Accessor3, 
          class Iterator4, class Accessor4, 
          class Functor>
inline void
inspectFourMultiArrays(Iterator1 s1, Shape const & shape, Accessor1 a1,
                      Iterator2 s2, Accessor2 a2,
                      Iterator3 s3, Accessor3 a3,
                      Iterator4 s4, Accessor4 a4,
                      Functor & f)
{
    inspectFourMultiArrays_binder<Iterator1, Shape, Accessor1,
                                 Iterator2, Accessor2,
                                 Iterator3, Accessor3, 
                                 Iterator4, Accessor4>
        g(s1, shape, a1, s2, a2, s3, a3, s4, a4);
    detail::extra_passes_select(g, f);
}
    
template <class Iterator1, class Shape, class Accessor1, 
          class Iterator2, class Accessor2, 
          class Iterator3, class Accessor3, 
          class Iterator4, class Accessor4, 
          class Functor>
inline void
inspectFourMultiArrays(triple<Iterator1, Shape, Accessor1> const & s1, 
                      pair<Iterator2, Accessor2> const & s2,  
                      pair<Iterator3, Accessor3> const & s3, 
                      pair<Iterator4, Accessor4> const & s4,
                      Functor & f)
{
    inspectFourMultiArrays(s1.first, s1.second, s1.third, 
                          s2.first, s2.second,
                          s3.first, s3.second, 
                          s4.first, s4.second,  f);
}

template <unsigned int N, class T1, class S1, 
                          class T2, class S2, 
                          class T3, class S3, 
                          class T4, class S4, 
          class Functor>
inline void
inspectFourMultiArrays(MultiArrayView<N, T1, S1> const & s1,
                       MultiArrayView<N, T2, S2> const & s2, 
                       MultiArrayView<N, T3, S3> const & s3, 
                       MultiArrayView<N, T4, S4> const & s4, Functor & f)
{
    vigra_precondition(s1.shape() == s2.shape(),
        "inspectTwoMultiArrays(): shape mismatch between inputs.");

    inspectFourMultiArrays(srcMultiArrayRange(s1), 
                          srcMultiArray(s2),  
                          srcMultiArray(s3), 
                          srcMultiArray(s4),
                          f);
}


}