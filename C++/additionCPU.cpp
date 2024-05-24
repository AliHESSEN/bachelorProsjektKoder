#include <iostream>
#include "additionCPU.h"

/**
 * @brief Performs image addition on the CPU using OpenCV's addWeighted function.
 *
 * This function takes four input images and performs a weighted addition on the CPU.
 * The processing time for the addition is measured and displayed. The original images
 * and the resulting image after addition are also displayed using OpenCV's imshow function.
 *
 * @param image1 The first input image (cv::Mat).
 * @param image2 The second input image (cv::Mat).
 * @param image3 The third input image (cv::Mat).
 * @param image4 The fourth input image (cv::Mat).
 */

void additionAlgoCPU(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& image3, const cv::Mat& image4)
{
    // CPU prosessering
    cv::Mat resultCpuAddition; // Matrise for å lagre resultatet av bildeaddisjonen
    cv::TickMeter timeCpuAddition; // Objekt for å måle tid

    // gjør addisjon på CPU med addWeighted
    timeCpuAddition.start(); // begynner å måle tiden 
    cv::addWeighted(image1, 1, image2, 1, 0.0, resultCpuAddition); // legger sammen image1 og image2 med vektene 0.25
    cv::addWeighted(resultCpuAddition, 0.5, image3, 0.5, 0.0, resultCpuAddition); // legger sammen forrige resultat med image3 med vektene 0.5
    cv::addWeighted(resultCpuAddition, 0.75, image4, 0.75, 0.0, resultCpuAddition); // legger sammen forrige resultat med image4 med vektene 0.75
    timeCpuAddition.stop(); // Stopp tidtaking

    // regner ut tidenn for addisjon på cpu og skriver den ut i sekunder
    double timeForCpuAdd = timeCpuAddition.getTimeMilli() * 1e-3; // bruker 1e-3 for å konvertere til sekunder
    std::cout << "Time spent by CPU for addition: " << timeForCpuAdd << " seconds\n" << std::endl; // skrierv ut tiden
    std::cout << "----------------------------------------------------------------------------------------" << std::endl; 

    // viser de originale bildene og resultatet
    cv::imshow("Original Image 1", image1); 
    cv::imshow("Original Image 2", image2); 
    cv::imshow("Original Image 3", image3); 
    cv::imshow("Original Image 4", image4); 
    cv::imshow("CPU-Processed Image (Addition)", resultCpuAddition); // viser resultatet etter addisjonen
    cv::waitKey(0); 
    cv::destroyAllWindows(); // lukker alt
}
