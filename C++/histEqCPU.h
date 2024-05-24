#pragma once

#include <opencv2/imgproc.hpp>  // Inkludering av bildebehandlingsfunksjoner som fargeomforming og histogramutjevning.
#include <opencv2/highgui.hpp>  // Inkludering av funksjoner for å vise bilder og håndtere brukerinteraksjon med OpenCV-vinduer.


void histogramEqualizationAlgoCPU(const cv::Mat& image1);