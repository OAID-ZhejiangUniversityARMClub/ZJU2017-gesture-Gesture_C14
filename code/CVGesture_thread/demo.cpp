#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <stdio.h>
#include <string>
#include <thread>

#define TEST 0

cv::Mat img, img_orig;
std::vector<cv::Rect> vec[5];
cv::CascadeClassifier Clf[5];
std::string str[5] = {"fist", "palm", "V", "good", "finger"};
const int size_min[5] = {30, 45, 25, 25, 30};
const int size_max[5] = {100, 100, 100, 100, 100};
const int para[5] = {1, 2, 1, 1, 2};

#if TEST==0
int flag_valid = 0;
void DrawText(cv::Mat& img,std::string text,int x, int y,cv::Scalar color)
{
    cv::putText(img,text.c_str(),cv::Point(x,y),cv::FONT_HERSHEY_SIMPLEX,0.8,color,2,1);
}

#else
int cnt_ges[5] = {0};

#endif

void fun(unsigned int i)
{
    
    Clf[i].detectMultiScale(img_orig, vec[i], 1.4, para[i], 0|CV_HAAR_SCALE_IMAGE, cv::Size(size_min[i],size_min[i]), cv::Size(size_max[i],size_max[i]));

#if TEST==0
    for(unsigned int j=0; j<vec[i].size(); j++)
    {
        flag_valid = 1;
        cv::putText(img,str[i],cv::Point(vec[i][j].x,vec[i][j].y),cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(0,0,255),2,1);
        cv::rectangle(img,cv::Point(vec[i][j].x,vec[i][j].y),cv::Point(vec[i][j].x+vec[i][j].width,vec[i][j].y+vec[i][j].height),cv::Scalar(0,255,255),1,8,0);
        printf("%s %d*%d  ", str[i].c_str(), vec[i][j].width, vec[i][j].height);
    }

#else
    if(vec[i].size())cnt_ges[i]+=1;

#endif
}


int main()
{
    for(unsigned int i=0; i<5; i++)
    {
        Clf[i].load(str[i]+".xml");
    }

#if TEST==0
    cv::VideoCapture cap(0);
    double time_past=0;
    int cnt_frame=0, cnt_valid=0;
    int64 time_abs=cv::getTickCount();
    double avg_fps=0;

    while(true)
    {
        int64 start = cv::getTickCount();
        cap>>img_orig;
        img_orig.copyTo(img);

        std::thread t0(fun, 0);
        std::thread t1(fun, 1);
        std::thread t2(fun, 2);
        std::thread t3(fun, 3);
        std::thread t4(fun, 4);
        
        t0.join();
        t1.join();
        t2.join();
        t3.join();
        t4.join();

        cnt_frame++;
        if(flag_valid == 1)
        {
            flag_valid = 0;
            printf("at Frame %d\n", cnt_frame);
            cnt_valid += 1;
            time_past += (cv::getTickCount() - start) / cv::getTickFrequency();
        }
        if(cnt_valid >= 10)
        {
            avg_fps = (double)cnt_frame / ((cv::getTickCount() - time_abs) / cv::getTickFrequency());
            printf("average fps: %5.2f, inference time: %f ms per frame\n", avg_fps, time_past/cnt_valid*1000);
            cnt_frame = 0;
            time_abs = cv::getTickCount();
            cnt_valid = 0;
            time_past = 0;
        }
        char fps_str[256] ;
        sprintf(fps_str,"%s %d","FPS : ",(int)avg_fps);
        DrawText(img,fps_str,10,50,cv::Scalar(0,255,0));

        cv::imshow("Gesture Recognition",img);
        if(cv::waitKey(1)==27)
        {
            break;
        }
    }

#else
    char path_str[256];
    printf("%10s%10s%10s%10s%10s%10s\n", "", "fist", "palm", "V", "good", "finger");
    for(unsigned int i=0; i<5; i++)
    {
        for(unsigned int j=0; j<5; j++)cnt_ges[j]=0;
        for(unsigned int j=0; j<300; j++)
        {
            sprintf(path_str, "/home/firefly/Desktop/train0/neg/%s/%04d.png", str[i].c_str(), j);
            img_orig = cv::imread(path_str);

            std::thread t0(fun, 0);
            std::thread t1(fun, 1);
            std::thread t2(fun, 2);
            std::thread t3(fun, 3);
            std::thread t4(fun, 4);
        
            t0.join();
            t1.join();
            t2.join();
            t3.join();
            t4.join();

            
        }
        printf("%10s%10d%10d%10d%10d%10d\n", str[i].c_str(), cnt_ges[0], cnt_ges[1], cnt_ges[2], cnt_ges[3], cnt_ges[4]);
    }

#endif

    return 0;
}

