 
 /************************************************************************/
 /*                                                                      */
 /*     Copyright 2006-2007 by F. Heinrich, B. Seppke, Ullrich Koethe    */
 /*                                                                      */
 /*    This file is part of the VIGRA computer vision library.           */
 /*    The VIGRA Website is                                              */
 /*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
 /*    Please direct questions, bug reports, and contributions to        */
 /*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
 /*        vigra@informatik.uni-hamburg.de                               */
 /*                                                                      */
 /*    Permission is hereby granted, free of charge, to any person       */
 /*    obtaining a copy of this software and associated documentation    */
 /*    files (the "Software"), to deal in the Software without           */
 /*    restriction, including without limitation the rights to use,      */
 /*    copy, modify, merge, publish, distribute, sublicense, and/or      */
 /*    sell copies of the Software, and to permit persons to whom the    */
 /*    Software is furnished to do so, subject to the following          */
 /*    conditions:                                                       */
 /*                                                                      */
 /*    The above copyright notice and this permission notice shall be    */
 /*    included in all copies or substantial portions of the             */
 /*    Software.                                                         */
 /*                                                                      */
 /*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
 /*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
 /*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
 /*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
 /*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
 /*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
 /*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
 /*    OTHER DEALINGS IN THE SOFTWARE.                                   */                
 /*                                                                      */
 /************************************************************************/
 
 #include <iostream>
 #include <functional>
 #include <cmath>
#include <boost/concept_check.hpp>
 #include "vigra/unittest.hxx"
 
#include "vigra/multi_array.hxx"
#include "vigra/union_find.hxx"
#include "mergeLabels.hxx"



struct MergeLabelTest
{
    typedef unsigned char PixelType;
    typedef unsigned int LabelType;
    
    typedef vigra::MultiArray<3,PixelType> Volume_t;
    typedef vigra::MultiArray<2,PixelType> Image_t;
    typedef vigra::MultiArray<3,LabelType> VLabel_t;
    typedef vigra::MultiArray<2,LabelType> ILabel_t;
    typedef vigra::MultiArray<1,LabelType> Map_t;
    
    Volume_t left, right;
    Image_t left2, right2;
    VLabel_t leftLabels, rightLabels;
    ILabel_t leftLabels2, rightLabels2;
    Map_t leftMap, rightMap;
    
    MergeLabelTest() :
    left(Volume_t::difference_type(4,1,1)), right(Volume_t::difference_type(4,1,1)),
    leftLabels(VLabel_t::difference_type(4,1,1)), rightLabels(VLabel_t::difference_type(4,1,1)),
    leftMap(Map_t::difference_type(4)), rightMap(Map_t::difference_type(4)),
    left2(Image_t::difference_type(4,1)), right2(Image_t::difference_type(4,1)),
    leftLabels2(ILabel_t::difference_type(4,1)), rightLabels2(ILabel_t::difference_type(4,1))
    {
        static const PixelType leftData[] = {0, 0, 1, 3};
        static const PixelType rightData[] = {0, 0, 2, 3};
        static const LabelType leftLabelData[] = {0, 1, 2, 3};
        static const LabelType rightLabelData[] = {0, 1, 2, 3};
        static const LabelType leftMapData[] = {0, 1, 2, 3};
        static const LabelType rightMapData[] = {0, 5, 6, 7};

        const PixelType * p = leftData;
        Image_t::iterator j = left2.begin();
        for(Volume_t::iterator i = left.begin(); i != left.end(); ++i, ++p, ++j)
        {
            *i=*p;
            *j=*p;
        }
        p = rightData;
        j = right2.begin();
        for(Volume_t::iterator i = right.begin(); i != right.end(); ++i, ++p, j++)
        {
            *i=*p;
            *j=*p;
        }
        
        const LabelType * q = leftLabelData;
        ILabel_t::iterator k = leftLabels2.begin();
        for(VLabel_t::iterator i = leftLabels.begin(); i != leftLabels.end(); ++i, ++q, k++)
        {
            *i=*q;
            *k=*q;
        }
        q = rightLabelData;
        k = rightLabels2.begin();
        for(VLabel_t::iterator i = rightLabels.begin(); i != rightLabels.end(); ++i, ++q, k++)
        {
            *i=*q;
            *k=*q;
        }
        
        const LabelType * r = leftMapData;
        for(Map_t::iterator i = leftMap.begin(); i != leftMap.end(); ++i, ++r)
        {
            *i=*r;
        }
        r = rightMapData;
        for(Map_t::iterator i = rightMap.begin(); i != rightMap.end(); ++i, ++r)
        {
            *i=*r;
        }
    
    }
    
    void mergeLabelTest3d()
    {
        vigra::detail::UnionFindArray<LabelType> uf(rightMap[3]+1);
        // we need to specify the exact template because MultiArray cannot be cast to MultiArrayView
        vigra::mergeLabels<3, PixelType, LabelType>(left, right, leftLabels, rightLabels, leftMap, rightMap, uf);
//         for (LabelType i=0; i<8; i++)
//             std::cerr << uf.find(i) << " ";
//         std::cerr << std::endl << std::endl;
        should(uf.find(5) == uf.find(1));
        should(uf.find(7) == uf.find(3));
        should(uf.find(6) != uf.find(2));
    }
    
    void mergeLabelTest2d()
    {
//         std::cerr << "=================" << std::endl;
//         printMat(left2);
//         printMat(right2);
//         printMat(leftLabels2);
//         printMat(rightLabels2);
//         std::cerr << "=================" << std::endl;
        vigra::detail::UnionFindArray<LabelType> uf(rightMap[3]+1);
        vigra::mergeLabels<2, PixelType, LabelType>(left2, right2, leftLabels2, rightLabels2, leftMap, rightMap, uf);
//         for (LabelType i=0; i<8; i++)
//             std::cerr << uf.find(i) << " ";
//         std::cerr << std::endl << std::endl;
        should(uf.find(5) == uf.find(1));
        should(uf.find(7) == uf.find(3));
        should(uf.find(6) != uf.find(2));
    }
    
    template <class Array>
    void printMat(Array a)
    {
        typedef typename Array::iterator Iterator;
        Iterator it = a.begin();
        Iterator end = a.end();
        std::cerr << "[";
        for (; it<end; ++it)
        {
            std::cerr << (int) *it << " ";
        }
        std::cerr << "]" << std::endl;
    }
};

struct MergeLabelTestMore
{
    typedef unsigned char PixelType;
    typedef unsigned int LabelType;
    
    typedef vigra::MultiArray<2,PixelType> Image_t;
    typedef vigra::MultiArray<2,LabelType> ILabel_t;
    typedef vigra::MultiArray<1,LabelType> Map_t;
    
    Image_t left, right;
    ILabel_t leftLabels, rightLabels;
    Map_t leftMap, rightMap;
    
    MergeLabelTestMore() :
    left(Image_t::difference_type(4,2,1)), right(Image_t::difference_type(4,2,1)),
    leftLabels(Image_t::difference_type(4,2,1)), rightLabels(Image_t::difference_type(4,2,1)),
    leftMap(Map_t::difference_type(5)), rightMap(Map_t::difference_type(5))
    {
        static const PixelType leftData[] = {0, 13, 14, 14,
                                             0, 14, 14, 13};
        static const PixelType rightData[] = {0,  0, 14,  0,
                                              0,  0,  0, 13};
        static const LabelType leftLabelData[] = {0, 2, 1, 1,
                                                  0, 1, 1, 3};
        static const LabelType rightLabelData[] = {0, 0, 2, 0,
                                                   0, 0, 0, 1};
        static const LabelType leftMapData[] = {0, 1, 2, 3, 4};
        static const LabelType rightMapData[] = {0, 5, 6, 7, 8};
        
        const PixelType * p = leftData;
        for(Image_t::iterator i = left.begin(); i != left.end(); ++i, ++p)
        {
            *i=*p;
        }
        p = rightData;
        for(Image_t::iterator i = right.begin(); i != right.end(); ++i, ++p)
        {
            *i=*p;
        }
        
        const LabelType * r = leftLabelData;
        for(ILabel_t::iterator i = leftLabels.begin(); i != leftLabels.end(); ++i, ++r)
        {
            *i=*r;
        }
        
        r = rightLabelData;
        for(ILabel_t::iterator i = rightLabels.begin(); i != rightLabels.end(); ++i, ++r)
        {
            *i=*r;
        }
        
        r = leftMapData;
        for(Map_t::iterator i = leftMap.begin(); i != leftMap.end(); ++i, ++r)
        {
            *i=*r;
        }
        r = rightMapData;
        for(Map_t::iterator i = rightMap.begin(); i != rightMap.end(); ++i, ++r)
        {
            *i=*r;
        }
        
    }
    
    void mergeLabelTest()
    {
        vigra::detail::UnionFindArray<LabelType> uf(rightMap[4]+1);
        // we need to specify the exact template because MultiArray cannot be cast to MultiArrayView
        vigra::mergeLabels<2, PixelType, LabelType>(left, right, leftLabels, rightLabels, leftMap, rightMap, uf);
//         for (LabelType i=0; i<rightMap[4]+1; i++)
//             std::cerr << uf.find(i) << " ";
//         std::cerr << std::endl << std::endl;
        static const LabelType a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        static const LabelType b[] = {0, 6, 2, 5, 4, 5, 6, 7, 8};
        
        for (int i=0; i < 9; i++)
        {
            should(uf.find(a[i]) == uf.find(b[i]));
        }
    }
    
    template <class Array>
    void printMat(Array a)
    {
        typedef typename Array::iterator Iterator;
        Iterator it = a.begin();
        Iterator end = a.end();
        std::cerr << "[";
        for (; it<end; ++it)
        {
            std::cerr << (int) *it << " ";
        }
        std::cerr << "]" << std::endl;
    }
};



struct MergeLabelTestSuite
: public vigra::test_suite
{
    MergeLabelTestSuite()
    : vigra::test_suite("MergeLabelTestSuite")
    {
        add( testCase( &MergeLabelTest::mergeLabelTest2d));
        add( testCase( &MergeLabelTest::mergeLabelTest3d));
        add( testCase( &MergeLabelTestMore::mergeLabelTest));
    }
};

int main(int argc, char ** argv)
{
    MergeLabelTestSuite test;
    
    int failed = test.run(vigra::testsToBeExecuted(argc, argv));
    
    std::cout << test.report() << std::endl;
    return (failed != 0);
}

