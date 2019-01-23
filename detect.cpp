#include "detect.h"
#include "kcftracker.hpp"
#include "DBSCAN.h"
#include "time.h"
//#include "HierarchicalCluster.h"
#define PI 3.1415926
static KCFTracker tracker(1,0,1,0);//create KCFTracker object
int otsu(Mat binary)		//大津法Otsu
{
	int histogram[256] = { 0 };
	int thresh=150;
	int gray_max, gray_min, i, j, size;
	int g_sum_min, g_sum_max, p_sum_min, p_sum_max;//存储前景、背景的灰度和与个数和
	double w0 = 0, w1 = 0;		 //前景和背景所占整幅图像的比例  
	double u0 = 0, u1 = 0;		 //前景和背景的平均灰度
	double u = 0;					//图像总平均灰度
	double variance = 0;
	double max_variance = 0;//最大类间方差
	Mat src;
	gray_max = 0;
	gray_min = 255;
	src = binary.clone();
		//cvtColor(src, src, CV_BGR2GRAY);
	Mat img = src.clone();
	size = (img.cols)*(img.rows);		//获得图像大小
	for (i = 0; i<img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			int x = img.at<uchar>(i, j);
			if (x > gray_max)
				gray_max = x;
			if (x < gray_min)
				gray_min = x;
			histogram[x]++;		//统计灰度图直方图
		}
	for (i = gray_min; i <= gray_max; i++)
	{
		g_sum_max = 0;
		g_sum_min = 0;
		p_sum_max = 0;
		p_sum_min = 0;
		w0 = 0;
		w1 = 0;
		for (j = gray_min; j < i; j++) {
			g_sum_min += j * histogram[j];
			p_sum_min += histogram[j];
		}
		u0 = (double)g_sum_min / p_sum_min;
		w0 = (double)p_sum_min / size;
		for (j = i; j <= gray_max; j++) {
			g_sum_max += histogram[j] * j;
			p_sum_max += histogram[j];
		}
		u1 = (double)g_sum_max / p_sum_max;
		w1 = 1 - w0;
		u = u0 * w0 + u1 * w1;
		variance = w0 * w1*(u0 - u1)*(u0 - u1);
		if (variance > max_variance) {
			max_variance = variance;
			thresh = i;
		}
	}
	return thresh;
}

void multi_seg(Mat src)//先利用Otsu算法将图片分为两类，取类内方差最大的继续使用Otsu算法分开，直到类间方差与图像总方差比值满足一定条件
{
	int thresh=0;
	int histogram[256] = { 0 };
	int threshs[256] = { 0 };
	int q, gray_max, gray_min, i, j, size;
	double gray_ave, vt;
	int g_sum_min, g_sum_max, p_sum_min, p_sum_max;
	double var_max = 0;//每一类的类内方差及最大方差
	double var_all;//类间方差
	double w0 = 0, w1 = 0; //前景和背景所占整幅图像的比例  
	double u0 = 0, u1 = 0; //前景和背景的平均灰度
	double u = 0;
	double variance = 0;
	double max_variance = 0;//最大类间方差
	double SF;
	gray_ave = 0;//图像的平均灰度值
	gray_max = 0;
	gray_min = 255;
	vt = 0;//vt为图像的总方差
	SF = 0;//SF为结束条件,SF=omeg/vt
	q = 1;
	//cvtColor(src, src, COLOR_BGR2GRAY);
	Mat img = src.clone();
	size = (img.cols)*(img.rows);
	for (i = 0; i < img.rows; i++) //获得灰度直方图
		for (j = 0; j < img.cols; j++) {
			int x = img.at<uchar>(i, j);
			if (x > gray_max)
				gray_max = x;
			if (x < gray_min)
				gray_min = x;
			histogram[x]++;
		}
	threshs[0] = gray_min;
	threshs[1] = gray_max;
	for (i = gray_min; i <= gray_max; i++)
		gray_ave += (histogram[i] * i);
	gray_ave /= size;
	for (i = gray_min; i <= gray_max; i++)
		vt = vt + histogram[i] * (i - gray_ave)*(i - gray_ave);
	vt /= size;
	while (SF<0.95)//SF>0.95认为分割较为完全
	{
		SF = 0;
		max_variance = 0;
		int isize = 0;
		for (i = gray_min; i <= gray_max; i++)
			isize += histogram[i];
		for (i = gray_min; i <= gray_max; i++)
		{
			g_sum_max = 0;
			g_sum_min = 0;
			p_sum_max = 0;
			p_sum_min = 0;
			w0 = 0;
			w1 = 0;
			for (j = gray_min; j < i; j++) {
				g_sum_min += j * histogram[j];
				p_sum_min += histogram[j];
			}
			u0 = (double)g_sum_min / p_sum_min;
			w0 = (double)p_sum_min / isize;
			for (j = i; j <= gray_max; j++) {
				g_sum_max += histogram[j] * j;
				p_sum_max += histogram[j];
			}
			u1 = (double)g_sum_max / p_sum_max;
			w1 = 1 - w0;
			u = u0 * w0 + u1 * w1;
			variance = w0 * w1*(u0 - u1)*(u0 - u1);
			if (variance > max_variance) {
				max_variance = variance;
				thresh = i;
			}
		}
		int x = 0;
		while (thresh >threshs[x] && x <= q)//直接插入排序
			x++;
		for (int y = q; y >= x; y--)
			threshs[y + 1] = threshs[y];
		threshs[x] = thresh;
		q++;
		var_max = 0;
		double omeg = 0;
		for (x = 0; x < q; x++) {//类内方差统计
			int y;
			int N = 0;
			double ave = 0;
			double var_local = 0;
			for (y = threshs[x]; y <= threshs[x + 1]; y++) {
				N += histogram[y];
				ave += y * histogram[y];
			}
			ave /= N;
			omeg += N * (ave - gray_ave)*(ave - gray_ave);
			for (y = threshs[x]; y <= threshs[x + 1]; y++) {
				var_local += (y - ave)*(y - ave)*histogram[y];

			}
			var_local /= N;
			if (var_local > var_max) {
				var_max = var_local;
				gray_min = threshs[x];
				gray_max = threshs[x + 1];
			}
		}
		omeg /= size;
		SF = omeg / vt;
	}
	//对图像进行分阈值分割
	uchar *a, *b;
	for (i = 0; i < src.rows; i++) {
		a = src.ptr<uchar>(i);
		b = img.ptr<uchar>(i);
		for (j = 0; j < src.cols; j++)
		{
			int s = 0;
			while (a[j] > threshs[s] && s < q)
				s++;
			b[j] = threshs[s + 1];
		}
	}
	imshow("multi_threshold", img);
	//imwrite("task6.jpg", img);
}


vector<Point2f> enemy_Dec::obvious_dec(Mat frame)//detecting enemy
{
	Mat I;
	cvtColor(frame,I,CV_BGR2GRAY);
	Mat planes[] = { Mat_<float>(I), Mat::zeros(I.size(), CV_32F) };
	Mat complexI;
	//构造复数双通道矩阵
	merge(planes,2,complexI);
	//快速傅里叶变换
	dft(complexI,complexI);
	Mat mag, pha, mag_mean;
	Mat Re, Im;
	//分离复数到实部和虚部
	Re = planes[0]; //实部
	split(complexI, planes); 
	Re = planes[0]; //实部
	Im = planes[1]; //虚部
	//计算幅值
	magnitude(Re, Im, mag); 
	//计算相角
	phase(Re, Im, pha); 
 
	float *pre, *pim, *pm, *pp;
	//对幅值进行对数化
	for (int i = 0; i<mag.rows; i++)
	{
		pm = mag.ptr<float>(i);
		for (int j = 0; j<mag.cols; j++)
		{
			*pm = log(*pm);
			pm++;
		}
	}
	//对数谱的均值滤波
	blur(mag, mag_mean, Size(5, 5)); 
	//求取对数频谱残差
	mag = mag - mag_mean; 
 
	for (int i = 0; i<mag.rows; i++)
	{
		pre = Re.ptr<float>(i);
		pim = Im.ptr<float>(i);
		pm = mag.ptr<float>(i);
		pp = pha.ptr<float>(i);
		for (int j = 0; j<mag.cols; j++)
		{
			*pm = exp(*pm);
			*pre = *pm * cos(*pp);
			*pim = *pm * sin(*pp);
			pre++;
			pim++;
			pm++;
			pp++;
		}
	}
	Mat planes1[] = { Mat_<float>(Re), Mat_<float>(Im) };
	//重新整合实部和虚部组成双通道形式的复数矩阵
	merge(planes1, 2, complexI); 
	// 傅立叶反变换
	idft(complexI, complexI, DFT_SCALE); 
	//分离复数到实部和虚部
	split(complexI, planes); 
	Re = planes[0];
	Im = planes[1];
	//计算幅值和相角
	magnitude(Re, Im, mag); 
	for (int i = 0; i<mag.rows; i++)
	{
		pm = mag.ptr<float>(i);
		for (int j = 0; j<mag.cols; j++)
		{
			*pm = (*pm) * (*pm);
			pm++;
		}
	}
	GaussianBlur(mag, mag, Size(3, 3),2, 2);//如何提高鲁棒性？？
	Mat invDFT, invDFTcvt;
	//归一化到[0,255]供显示
	normalize(mag, invDFT, 0, 255, NORM_MINMAX); 
	//normalize(mag,invDFT,0,1.0,NORM_MINMAX);
	//转化成CV_8U型
	
	invDFT.convertTo(invDFTcvt, CV_8U);
	int thres=otsu(invDFTcvt);//可考虑采用不同的阈值选择方式!
	threshold(invDFTcvt,invDFTcvt,thres,255,THRESH_BINARY);
	//转化为二维点集
	vector<Point2f> points;
	int k=0;
	for(int i=0;i<invDFTcvt.rows;i++){
			const uchar* p=invDFTcvt.ptr<uchar>(i);
			for(int j=0;j<invDFTcvt.cols;j++){
				if(p[j]==255){
					k++;
					points.push_back(Point2f(j*1.0,i*1.0));
				}
			}
		}
	//invDFT.convertTo(invDFTcvt, CV_32FC1);  
	imshow("SpectualResidual", invDFTcvt);
	//imshow("Original Image", I);
	//waitKey(0);
	return points;
}
/*obv_dec:the gray image that has been dectected*/
int enemy_Dec::fir_center_dec(vector<Point2f> points)
//考虑先采用k-means或其他聚类方法找出目标中心，再利用meanshift或KCF对目标进行动态跟踪
{
/*-------------------------------------Using K-means to find out the center-------------------------------------------*/
/*	//容易受边缘噪点干扰
	size_t i,j,k=0;
	Mat samples(points.size(),1,CV_32FC2);
	Mat lables;
	Mat center(1,3,samples.type());//3
	for(i=0;i<points.size();i++){
		samples.at<Point2f>(i)=points[i];
		}
	kmeans(samples,3,lables,TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),2,KMEANS_PP_CENTERS, center);
	if(!center.empty()){
		for(i=0;i<1;i++)
			centers.push_back(Point(center.at<float>(i*2),center.at<float>(i*2+1)));
		//is_detected=true;
		selection=Rect(centers[1].x-50,centers[1].y-50,100,100);
		return 1;
		}
	else
		return 0;*/

/*---------------------Using meanshift to detect the center-----------------------------------------------------------*/
/*---------------------Using DBSCAN to detect the center---------------------------------------------*/
	DBSCAN(points,0.8,50);//cost too much time
	waitKey(10);
}


vector<float> enemy_Dec::angle_get(Mat frame)//基于检测到的中心点反馈小车要转动的角度
{
	vector<float> angles;
	double camD[9]={1,2,3,4,5,6,7,8,9};//内参矩阵
	Mat cam_matrix=Mat(3,3,CV_64FC1,camD);
	Mat N=cam_matrix.inv();//逆矩阵
	Mat A=Mat(3,1,CV_64FC1);
	Mat B=Mat(3,1,CV_64FC1);
	A.at<float>(0)=frame.cols/2;
	A.at<float>(1)=frame.rows/2;
	A.at<float>(2)=1.0;
	float X1,Y1,Z1;
	float temp1,temp2;
	Mat NA=N*A;
	X1=NA.at<float>(0);
	Y1=NA.at<float>(1);
	Z1=NA.at<float>(2);
	temp1=sqrt(Y1*Y1+Z1*Z1);
	atan2(temp1,X1);
	float X2,Y2,Z2;
	for(int i=0;i<centers.size();i++){
		B.at<float>(0)=centers[i].x;
		B.at<float>(1)=centers[i].y;
		B.at<float>(2)=1.0;
		Mat NB=N*B;
		X2=NB.at<float>(0);
		Y2=NB.at<float>(1);
		Z2=NB.at<float>(2);
		temp2=sqrt(Y2*Y2+Z2*Z2);
		atan2(temp2,X2);
		angles.push_back((temp2-temp1)/PI*180);//angles>0,turn right;else turn left
	}
	return angles;
}
		
void enemy_Dec::init()
{
	//tracker(HOG,FIXDWINDOW,MULTISCALE,LAB);
}

int enemy_Dec::process(Mat frame)
{
	if(trackCount<400)//此前未检测到目标
	{
		cout<<frame.rows<<endl;
		cout<<frame.cols<<endl;
		clock_t start=clock();
		vector<Point2f> points=obvious_dec(frame);
		//trackCount=fir_center_dec(points);
		centers.clear();
		fir_center_dec(points);
		trackCount++;
		//Point center=fir_center_dec(result);
		for(size_t i=0;i<centers.size();i++){
			circle(frame,centers[i],20,Scalar(0,0,255));
			}
		imshow("centers",frame);
		clock_t end=clock();
		float t=end-start;
		t/=1000000;
		cout<<t<<endl;
		waitKey(1);
	}
	else{
//此前已检测到目标//考虑改用KCF
/*-------------------------using meanshift to track--------------------------------------------------------------*/
/*		int smin=30,vmin=10,vmax=256;//adjustable!!!
		int binNum=16;//直方图分成多少个区间
		float hranges[]={0,180};
		const float* phranges=hranges;//统计像素的区间
		Mat hsv_img;
		cvtColor(frame,hsv_img,CV_BGR2HSV);
		//Using inRange to restrict S and V to filter some annoys
		inRange(hsv_img,Scalar(0,smin,min(vmin,vmax)),Scalar(180,256,max(vmin,vmax)),mask);
		int ch[]={0,0};
		hue.create(hsv_img.size(),hsv_img.depth());//hue初始化为与hsv大小深度一样的矩阵
		mixChannels(&hsv_img,1,&hue,1,ch,1);//将hsv第一个通道(也就是色调)的数复制到hue中
		if(trackCount==40){//对起始帧进行初始化
			histimg=Mat::zeros(frame.size(),frame.type());
			histimg=Scalar::all(0);
			Mat roi(hue,selection),maskrio(mask,selection);
			calcHist(&roi,1,0,maskroi,hist,1,&binNum,&phranges);
			normalize(hist,hist,0,255,CV_MINMAX);
			trackCount++;
		}
		calcBackProject(&hue,1,0,hist,backproj,&phranges);//计算直方图的反向投影，计算hue图像0通道直方图hist的反向投影，并存入backproj
		
		backproj &=mask;
		meanShift(backproj,selection,TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
		rectangle(frame,Point(selection.x,selection.y),Point(selection.x+selection.width,selection.y+selection.height),Scalar(0,0,255),1,CV_AA);
		trackCount++;
		imshow("target",frame);
		waitKey(100);*/
/*-----------------------Using KCF to track----------------------------------------------------------------------------*/
		
		if(trackCount==400){
			tracker.init(selection,frame);
			rectangle(frame,Point(selection.x,selection.y),Point(selection.x+selection.width,selection.y+selection.height),Scalar(0,0,255),1,CV_AA);
			}
		else{
		selection=tracker.update(frame);
		rectangle(frame,Point(selection.x,selection.y),Point(selection.x+selection.width,selection.y+selection.height),Scalar(0,0,255),1,CV_AA);
		}
		if(!SILENT){
			imshow("tracing",frame);
			trackCount++;
			waitKey(5);
		}
/*---------------------------------------------------------------------------------------------------------------------*/
	}
	return 0;
}

int main()
{
	VideoCapture cap;
	Mat frame;
	cap.open("test.mp4");
	enemy_Dec test;
	//test.init();
	while(1){
		cap>>frame;
		test.process(frame);
	}	
	return 0;
}
		
