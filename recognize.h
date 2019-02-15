#ifndef RECOGNIZE_H
#define RECOGNIZE_H

#include "head.h"
#define SQURE_HEIGHT 100
using namespace cv;
using namespace std;

Mat multi_seg(Mat src);
int ransac(vector<Vec4i> lines,vector<Vec4i> &result);
void unevenLightCompensate(Mat &image, int blockSize);
int speBinarizeMethod(const Mat src,Mat &dest);

class recognize{
public:
float edge_dec(Mat frame);
private:
Mat element=getStructuringElement(MORPH_CROSS,Size(4,4));


};







#endif
