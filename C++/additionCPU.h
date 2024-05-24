#pragma once

#include <opencv2/core.hpp>  // Inneholder grunnleggende definisjoner og funksjoner for openCV
#include <opencv2/imgproc.hpp>  // For bildebehandlingsoperasjoner som addWeighted.
#include <opencv2/highgui.hpp> // For GUI-relaterte operasjoner som imshow.

void additionAlgoCPU(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& image3, const cv::Mat& image4);


