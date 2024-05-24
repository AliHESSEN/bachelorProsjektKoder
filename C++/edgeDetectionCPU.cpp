#include <iostream>
#include "edgeDetectionCPU.h"

/**
 * @brief Performs edge detection on an input image using CPU.
 *
 * This function takes an input image and performs edge detection on it
 * using CPU. It measures the time taken for the edge detection
 * and displays the original image along with the processed results.
 *
 * @param image1 The input image for edge detection.
 *
 * @details The function follows these steps:

 * - Converts the input image to grayscale on CPU using OpenCV's cvtColor function.
 * - Performs edge detection on CPU using OpenCV's Canny function.
 * - Measures the time taken for CPU edge detection.
 * - Displays the original image and the processed results using OpenCV's imshow function.

 */

void edgeDetectionAlgoCPU(const cv::Mat& image1)
{
    // CPU prosessering
    cv::Mat resultCpuEdge; // mat variabel som skal brukes til å lagre resultatet etter kantdeteksjon
    cv::TickMeter timeCpuEdge; // skal brukes til å måle tiden det tar

    
    timeCpuEdge.start(); // begynner å måle tiden
    cv::cvtColor(image1, resultCpuEdge, cv::COLOR_BGR2GRAY); // konverterer til svart-hvitt tatt fra chatgpt
    cv::Canny(resultCpuEdge, resultCpuEdge, 50, 150); // ggjør kantdeteksjonen, trenger ikke definere kjernestørrelse for sobel som på GPU, her bruker den 3x3
    timeCpuEdge.stop(); // stopper måling av tiden

    
    double timeForCpuEdge = timeCpuEdge.getTimeMilli() * 1e-3; // gjør om tiden til sekunder 1e-3
    std::cout << "Time spent by CPU for edge detection: " << timeForCpuEdge << " seconds" << std::endl; 
    std::cout << "----------------------------------------------------------------------------------------" << std::endl;

    
    cv::imshow("Original Image", image1); // viser bildet som ble brukt 
    cv::imshow("CPU-Processed Image (Edge Detection)", resultCpuEdge); // viser resultatet etter kantdeteksjon
    cv::waitKey(0); 
    cv::destroyAllWindows(); 
}
