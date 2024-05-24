#pragma once


#include <opencv2/core.hpp> // Inneholder grunnleggende definisjoner og funksjoner.
#include <opencv2/core/cuda.hpp> // For CUDA-relaterte definisjoner og operasjoner.
#include <opencv2/cudaarithm.hpp> // For aritmetiske operasjoner på CUDA.
#include <opencv2/highgui.hpp> // For GUI-relaterte operasjoner som imshow.
#include <opencv2/imgproc.hpp> // For bildebehandlingsoperasjoner som addWeighted.


void additionAlgoGPU(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& image3, const cv::Mat& image4);

