 
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
 #include "vigra/unittest.hxx"
 
#include "vigra/multi_array.hxx"
#include "vigra/union_find.hxx"
#include "mergeLabels.hxx"



struct MergeLabelTest
{
    typedef unsigned char PixelType;
    typedef unsigned int LabelType;
    
    typedef vigra::MultiArray<3,PixelType> Volume_t;
    typedef vigra::MultiArray<3,LabelType> Label_t;
    typedef vigra::MultiArray<1,LabelType> Map_t;
    
    Volume_t left, right;
    Label_t leftLabels, rightLabels;
    Map_t leftMap, rightMap;
    
    MergeLabelTest()
    : left(Volume_t::difference_type(4,1,1)), right(Volume_t::difference_type(4,1,1)),
    leftLabels(Label_t::difference_type(4,1,1)), rightLabels(Label_t::difference_type(4,1,1)),
    leftMap(Map_t::difference_type(4)), rightMap(Map_t::difference_type(4))
    {
        static const PixelType leftData[] = {0, 0, 1, 3};
        static const PixelType rightData[] = {0, 0, 2, 3};
        static const LabelType leftLabelData[] = {0, 1, 2, 3};
        static const LabelType rightLabelData[] = {0, 1, 2, 3};
        static const LabelType leftMapData[] = {0, 1, 2, 3};
        static const LabelType rightMapData[] = {0, 5, 6, 7};

        const PixelType * p = leftData;
        for(Volume_t::iterator i = left.begin(); i != left.end(); ++i, ++p)
        {
            *i=*p;
        }
        p = rightData;
        for(Volume_t::iterator i = right.begin(); i != right.end(); ++i, ++p)
        {
            *i=*p;
        }
        
        const LabelType * q = leftLabelData;
        for(Label_t::iterator i = leftLabels.begin(); i != leftLabels.end(); ++i, ++q)
        {
            *i=*q;
        }
        q = rightLabelData;
        for(Label_t::iterator i = rightLabels.begin(); i != rightLabels.end(); ++i, ++q)
        {
            *i=*q;
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
    
    void mergeLabelTest()
    {
        vigra::detail::UnionFindArray<LabelType> uf(rightMap[3]+1);
        vigra::mergeLabels<3, PixelType, LabelType>(left, right, leftLabels, rightLabels, leftMap, rightMap, uf);
        for (LabelType i=0; i<8; i++)
            std::cerr << uf.find(i) << " " << std::endl;
        should(uf.find(5) == uf.find(1));
        should(uf.find(7) == uf.find(3));
        should(uf.find(6) != uf.find(2));
    }
    
    
};



struct MergeLabelTestSuite
: public vigra::test_suite
{
    MergeLabelTestSuite()
    : vigra::test_suite("MergeLabelTestSuite")
    {
        add( testCase( &MergeLabelTest::mergeLabelTest));
    }
};

int main(int argc, char ** argv)
{
    MergeLabelTestSuite test;
    
    int failed = test.run(vigra::testsToBeExecuted(argc, argv));
    
    std::cout << test.report() << std::endl;
    return (failed != 0);
}

