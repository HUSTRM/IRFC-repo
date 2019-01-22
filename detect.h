#ifndef DETECT_H
#define DETECT_H

#include "head.h"
using namespace cv;
using namespace std;

int otsu(Mat frame);
/*-----------------------------------------------------------------------------------------------*/
class enemy_Dec
{
private:
//持续性检测过程中，当灰度传感器检测到边缘后，将被置为false，即已将对方推下去
Mat histimg;
vector<Point2f> centers;//如果有多个中心
int trackCount=0;//if it has detected the target for the first time
Mat hue, mask, hist, histImg,backproj,maskroi;
Rect selection;
bool HOG=true;
bool FIXDWINDOW=false;
bool MULTISCALE=true;
bool SILENT=false;
bool LAB=false;
public:
vector<Point2f> obvious_dec(Mat frame);
int process(Mat frame);//detecting enemy
int fir_center_dec(vector<Point2f> points);
vector<float> angle_get(Mat frame);
void init();
};

/*----------------------------------------------------------------------------------------------*/

#endif
