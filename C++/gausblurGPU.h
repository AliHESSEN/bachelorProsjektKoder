#pragma once

#include <opencv2/highgui.hpp>   // For vindush�ndtering og GUI-funksjoner som imshow
#include <opencv2/core/cuda.hpp> // Grunnleggende st�tte for CUDA-akselerasjon
#include <opencv2/cudafilters.hpp> // For GPU-basert filtre, inkludert Gaussian Blur
#include <opencv2/cudaimgproc.hpp> // Hvis du trenger ekstra bildemanipulasjoner p� GPU


void gaussianBlurAlgoGPU(const cv::Mat& image);
