#include <iostream>
#include "subtractionCPU.h"

/**
 * @brief Performs subtraction operation on two images using both GPU and CPU.
 *
 * This function takes two input images and performs subtraction operation on them
 * using CPU. It measures the time taken for the subtraction operation
 * and displays the original images along with the processed results.
 *
 * @param image1 The first input image.
 * @param image2 The second input image.
 *
 * @details The function follows similar steps as addition, but uses OpenCV's subtract function.
 */

void subtractionAlgoCPU(const cv::Mat& image1, const cv::Mat& image2)
{
    

    cv::Mat resultCpuSubtraction; // varibelen her skal bruke for å holde resultatet etter subtraksjon
    cv::TickMeter cpuTime; // varibel for å måle tider

    
    cpuTime.start(); // Starter timeren
    cv::subtract(image1, image2, resultCpuSubtraction); // kjører subtraksjon på CPU
    cpuTime.stop(); // stopper tida

    // konverterer tider til sekunder 1e-3
    double usedTimeCpuSubtraction = cpuTime.getTimeMilli() * 1e-3; 
    std::cout << "Time spent by CPU for subtraction: " << usedTimeCpuSubtraction << " seconds" << std::endl; // tiden skrives ut
    std::cout << "----------------------------------------------------------------------------------------" << std::endl;

    
    cv::imshow("Original Image 1", image1); // det første originalbilde som ble brukt
    cv::imshow("Original Image 2", image2); // også viser vi det andre originalbilde som ble brukt
    cv::imshow("CPU-Processed Image (Subtraction)", resultCpuSubtraction); // resultatet blir vist her
    cv::waitKey(0); 
    cv::destroyAllWindows(); 
}
