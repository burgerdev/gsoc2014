#include "vigra/multi_array.hxx"
#include "vigra/union_find.hxx"
#include "vigra/timing.hxx"
#include "mergeLabels.hxx"

#define N 4096

int main()
{
    typedef unsigned char PixelType;
    typedef unsigned int LabelType;
    
    typedef vigra::MultiArray<2,PixelType> Volume;
    typedef vigra::MultiArray<2,LabelType> Label;
    typedef vigra::MultiArray<1,LabelType> Map;

    vigra::detail::UnionFindArray<LabelType> uf((LabelType)1);
    
    Volume left(Volume::difference_type(N,N), (PixelType)0);
    Volume right(Volume::difference_type(N,N), (PixelType)0);
    
    Label leftLabels(Label::difference_type(N,N), (LabelType)0);
    Label rightLabels(Label::difference_type(N,N), (LabelType)0);
    
    Map leftMap(Map::difference_type(1), (LabelType)0);
    Map rightMap(Map::difference_type(1), (LabelType)0);
    
    USETICTOC
    
    TIC
    vigra::mergeLabels<2, PixelType, LabelType>(left, right, leftLabels, rightLabels, leftMap, rightMap, uf);
    TOC
    
    TIC
    vigra::mergeLabelsRaw<2, PixelType, LabelType>(left, right, leftLabels, rightLabels, leftMap, rightMap, uf);
    TOC
    
}