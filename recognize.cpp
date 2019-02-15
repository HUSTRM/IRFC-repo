#include "recognize.h"
//思路：对不均匀光照进行补偿，再结合滤波 边缘检测 以及霍夫变换等算法检测并筛选出合适的线条

int hou_thresh=22,hou_length=120,hou_gap=15,mul_thres=75,low_thres=30,high_thres=60;
Mat dele,FRAME;
Mat multi_seg(Mat src)//先利用Otsu算法将图片分为两类，取类内方差最大的继续使用Otsu算法分开，直到类间方差与图像总方差比值满足一定条件
{
	int histogram[256] = { 0 };
	int threshs[256] = { 0 };
	int thresh;
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
	float t_thres=static_cast<float>(mul_thres)/100.0;
	while (SF<t_thres)//SF>0.95认为分割较为完全
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
	imshow("gray", img);
	return img;
}


void unevenLightCompensate(Mat &image, int blockSize)
{
	if (image.channels() == 3) cvtColor(image, image, 7);//gray
	double average = mean(image)[0];
	int rows_new = ceil(double(image.rows) / double(blockSize));
	int cols_new = ceil(double(image.cols) / double(blockSize));
	Mat blockImage;
	blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i*blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > image.rows) rowmax = image.rows;
			int colmin = j*blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > image.cols) colmax = image.cols;
			Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
			double temaver = mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	Mat blockImage2;
	resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
	Mat image2;
	image.convertTo(image2, CV_32FC1);
	Mat dst = image2 - blockImage2;
	dst.convertTo(image, CV_8UC1);
}


//void nms(const vector<

int ransac(vector<Vec4i> lines,vector<Vec4i> &result)//对检测到的线条用ransac进行筛选
{
	vector<Point2f> p1(lines.size());
	vector<Point2f> p2(lines.size());
	for(size_t i=0;i<lines.size();i++){
		p1[i]=Point(static_cast<float>(lines[i][0]),static_cast<float>(lines[i][1]));
		p2[i]=Point(static_cast<float>(lines[i][2]),static_cast<float>(lines[i][3]));
	}
	int ransac_rej=12;//reject threshold
	Mat H12;
	H12=findHomography(Mat(p1),Mat(p2),CV_RANSAC,ransac_rej);

/*in some cases,it will return an empty matrix.Therefore it's unable to using perspectivetransform to apply ransac*/
	if(!H12.empty()){
		vector<char> matchesMask(lines.size(),0);
		Mat p1_t;
		perspectiveTransform(Mat(p1),p1_t,H12);
		int j;
		for(j=0;j<lines.size();j++){
			if(norm(p2[j]-p1_t.at<Point2f>(j,0))<=ransac_rej){
				matchesMask[j]=1;
			}
		}
		for(j=0;j<lines.size();j++){
			if(matchesMask[j]==1){//inside point
				result.push_back(lines[j]);
			}
		}
		imshow("lines",FRAME);
		return 1;//success
	}
	else if(lines.size()>=8){
		vector<uchar> m_RANSACStatus;
		Mat m_Fundamental;
		m_Fundamental=findFundamentalMat(p1,p2,m_RANSACStatus,CV_FM_RANSAC);//N shoule be >=8
		for(int j=0;j<lines.size();j++){
			if(m_RANSACStatus[j]!=0){
				result.push_back(lines[j]);
			}
		}
		imshow("lines",FRAME);
		return 1;//success
	}
	else{
		result=lines;
		return -1;//failure
	}
}


/*功能说明：
小车掉落后检测赛场边缘帮助小车重回擂台
结合传感器
*/

float recognize::edge_dec(Mat frame)//detecting the edge to help the car to get to the main land
{
/*	Mat roi;
	Rect rect(0,frame.rows-SQURE_HEIGHT,frame.cols,SQURE_HEIGHT);//cut part of the image to run edge_dec
	frame(rect).copyTo(roi);
	Mat gray=roi.clone();*/
	Mat gray=frame.clone();
	GaussianBlur(gray,gray,Size(3,3),0,0);//de noise
	Mat temp0=gray.clone();
	gray=multi_seg(temp0);
	Mat temp;
	Canny(gray,temp,low_thres,high_thres);//adjust low_thres high_thres
	vector<Vec4i> lines,result;
	HoughLinesP(temp,lines,1.0,CV_PI/180,hou_thresh,hou_length,hou_gap);//threshold minLineLength maxLineGap

	Mat temp_p=FRAME.clone();
	for(size_t i=0;i<lines.size();i++){
		line(temp_p,Point(lines[i][0],lines[i][1]),Point(lines[i][2],lines[i][3]),Scalar(0,0,255),3,8);
	}
	imshow("before_ransac",temp_p);
	int p=ransac(lines,result);//p==1:successfully ransac;else failed
	//Non-Maximum-Suppression
	if(p==1)
		cout<<"success"<<endl;
	else{
		cout<<"fail"<<endl;
		cout<<result.size()<<" lines are detected"<<endl;
	}
	int j,temp_tag=1,k;
	vector<float> ks;
	vector<int> tags;
	float k_temp;
	float t_y,t_x;
	t_y=result[0][3]-result[0][1];
	t_x=result[0][2]-result[0][0];
	if((t_y>0&&t_x>0)||(t_y<0&&t_x<0)){//restrict the angle to 0-PI
		t_y=abs(t_y);
		t_x=abs(t_x);
	}
	else{
		t_x=-1*abs(t_x);
		t_y=abs(t_y);
	}
	k_temp=atan2(t_y,t_x)*180/CV_PI;
	ks.push_back(k_temp);
	tags.push_back(temp_tag);
	for(j=1;j<result.size();j++){//calculate k of the detected lines
		t_y=result[j][3]-result[j][1];
		t_x=result[j][2]-result[j][0];
		if((t_y>0&&t_x>0)||(t_y<0&&t_x<0)){
			t_y=abs(t_y);
			t_x=abs(t_x);
		}
		else{
			t_x=-1*abs(t_x);
			t_y=abs(t_y);
		}
		k_temp=atan2(t_y,t_x)*180/CV_PI;
		int p=0;
		for(k=0;k<ks.size();k++){
			if(fabs(k_temp-ks[k])<3.0){//seems like the same line
				tags[k]++;
				p=1;
				break;
			}
		}
		if(!p){
			temp_tag=1;
			tags.push_back(temp_tag);
			ks.push_back(k_temp);
		}
	}
	temp_tag=0;
	int max_tag=tags[0];
	for(j=0;j<tags.size();j++){
		if(tags[j]>max_tag){
			max_tag=tags[j];
			temp_tag=j;
		}
	}
	cout<<"the k of edge is "<<ks[temp_tag]<<" degree"<<endl;
	for(j=0;j<result.size();j++){
		line(FRAME,Point(result[j][0],result[j][1]),Point(result[j][2],result[j][3]),Scalar(0,0,255),3,8);
	}
	imshow("lines",FRAME);
	return 0.5;//???
}

int speBinarizeMethod(const Mat src,Mat &dest)
{
	if (src.empty()||src.channels()!=3)
		return 0;
	float laplcian_like_kernel[81]={0,0,0,0,-1,0,0,0,0,
					0,0,0,0,-1,0,0,0,0,
					0,0,0,0,-1,0,0,0,0,
					0,0,0,0,-1,0,0,0,0,
					-1,-1,-1,-1,16,-1,-1,-1,-1,
					0,0,0,0,-1,0,0,0,0,
					0,0,0,0,-1,0,0,0,0,
					0,0,0,0,-1,0,0,0,0,
					0,0,0,0,-1,0,0,0,0};
	float med_blur_kernel[10] = {1,1,1,1,1,1,1,1,1,1};
	Mat laplcianLkernel = Mat(9, 9, CV_32FC1, laplcian_like_kernel);
	Mat medblurKernel = Mat(5, 2, CV_32FC1, med_blur_kernel);
	Mat srcClone = src.clone();
	Mat grayImg, lSrcClone;
	Mat gaussKernel = getGaussianKernel(5, 2);
	GaussianBlur(srcClone, srcClone, Size(5,5),0,2);
	filter2D(srcClone, lSrcClone, srcClone.depth(), laplcianLkernel);
	cvtColor(lSrcClone, grayImg, CV_RGB2GRAY);
	medianBlur(grayImg, grayImg, 3);
	threshold(grayImg, dest, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);
	return 1;
}
	

static void dec_test(int,void *)
{
	FRAME=imread("lines.jpg");
	Mat temp=FRAME.clone();
/*
	vector<Mat> rgb;
	split(temp,rgb);
	for(int i=0;i<3;i++)
		equalizeHist(rgb[i],rgb[i]);
	merge(rgb,temp);
	vector<Mat> splited;
	cvtColor(temp,temp,CV_BGR2HSV);
	split(temp,splited);
	addWeighted(splited[2],1,splited[1],-1,0,dele);
	unevenLightCompensate(dele,35);
*/
	speBinarizeMethod(FRAME,dele);
	//imshow("Result",dele);
	recognize test;
	test.edge_dec(dele);
}
/*
在运动过程中检测目标侧面（基于轮胎）
enemy为检测或追踪过程中提取的目标框部分，对其进行轮廓检测判断是否为侧面
*/
/*
int recognize::side_dec(Mat enemy)
{
*/


int main()
{
	namedWindow("lines");
	namedWindow("gray");
	namedWindow("canny");
	createTrackbar("hough thres:","lines",&hou_thresh,255,dec_test);
	createTrackbar("hough min_length:","lines",&hou_length,200,dec_test);
	createTrackbar("hough max_gap","lines",&hou_gap,100,dec_test);
	createTrackbar("multiseg ","gray",&mul_thres,100,dec_test);
	createTrackbar("min_thres ","canny",&low_thres,100,dec_test);
	createTrackbar("max_thres ","canny",&high_thres,200,dec_test);
	dec_test(0,0);

	waitKey(0);
	return 0;
}

	
