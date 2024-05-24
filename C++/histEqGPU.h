#pragma once

#include <opencv2/core.hpp> // For grunnleggende OpenCV-funksjoner og klasser som cv::Mat.
#include <opencv2/core/cuda.hpp> // For CUDA-relaterte funksjoner og klasser som cv::cuda::GpuMat.
#include <opencv2/cudaarithm.hpp> // For aritmetiske operasjoner og funksjoner på GPU, som cv::cuda::equalizeHist.
#include <opencv2/cudaimgproc.hpp> // For bildebehandlingsfunksjoner på GPU, som cv::cuda::cvtColor.
#include <opencv2/highgui.hpp> // For å vise bilder med cv::imshow eller håndtere vinduer.



void histogramEqualizationAlgoGPU(const cv::Mat& image1);



