#include <iostream>
#include "edgeDetectionGPU.h"

/**
 * @brief Performs edge detection on an input image using GPU.
 *
 * This function takes an input image and performs edge detection on it using GPU. It measures the time taken for the edge detection
 * and displays the original image along with the processed results.
 *
 * @param image1 The input image for edge detection.
 *
 * @details
 * The function follows these steps:
 * - Uploads the input image to GPU memory.
 * - Converts the image to grayscale on GPU using OpenCV's cuda::cvtColor function.
 *   - Converting the image to grayscale simplifies the computation and enhances edge detection effectiveness,
 *     as Canny edge detection focuses on intensity gradients.
 * - Performs edge detection on GPU using OpenCV's cuda::CannyEdgeDetector.
 * - Measures the time taken for GPU edge detection.
 * - Downloads the result back to CPU memory.
 * - Displays the original image and the processed results using OpenCV's imshow function.
 *
 * @note The input image is expected to be in BGR format.
 *
 * @param image1 The input image for edge detection.
 */

void edgeDetectionAlgoGPU(const cv::Mat& image1)
{
    // GPU
    cv::cuda::GpuMat gpuImage1, gpuResultEdge; // lager GPU matriser en for input og en som skal brukes til resultat
    cv::cuda::Event startGpuUploadEdge, endGpuUploadEdge; // lager events som skal brukes til å måle opplastings tid til GPU
    startGpuUploadEdge.record(); // starter tiden
    gpuImage1.upload(image1); // laster opp til GPU minne
    endGpuUploadEdge.record(); // stopper målingen

    
    cv::cuda::cvtColor(gpuImage1, gpuImage1, cv::COLOR_BGR2GRAY); // gjør om til svart-hvitt bilde
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(50.0, 150.0, 3, false); // lager kantdetektor med canny som skal brukes for kantdeteksjonen. med grenseverider, og kjernestørrelse på 3
    cv::cuda::Event startGpuEdge, endGpuEdge; // lager events som skal bruke til å måle selve kantdeteksjonen

   
    startGpuEdge.record(); // starter målingen av tid
    canny->detect(gpuImage1, gpuResultEdge); // gjør kantdeteksjon med canny kantdetektoren som vi lagde tidligere på GPU
    endGpuEdge.record(); // stopper målingen
    endGpuEdge.waitForCompletion(); // sørger for at kantdeteksjonen skal er ferdig

    //  alle tider konverteres til sekunder ved 1e-3
    float timeForGpuUploadEdge = cv::cuda::Event::elapsedTime(startGpuUploadEdge, endGpuUploadEdge) * 1e-3; // variabel for opplastnings tider
    float timeForGpuEdge = cv::cuda::Event::elapsedTime(startGpuEdge, endGpuEdge) * 1e-3; // varbel for selve kantdeteksjonen
    std::cout << "----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\nTime spent sending data to GPU: " << timeForGpuUploadEdge << " seconds" << std::endl; 
    std::cout << "Time spent by GPU for edge detection: " << timeForGpuEdge << " seconds" << std::endl; 

    // lager CPU matrise som skal holde resultatet fra GPU 
    cv::Mat resultGpuEdge; // variabel for resultat bildet
    cv::cuda::Event startDataFromGpuToCpu, endDataFromGpuToCpu; // events som jeg skal bruke til å måle nedlastings tid
    startDataFromGpuToCpu.record(); // starter målingen
    gpuResultEdge.download(resultGpuEdge); // laster ned til CPU
    endDataFromGpuToCpu.record(); // stopper målingen

    float timeForDataFromGpuToCpu = cv::cuda::Event::elapsedTime(startDataFromGpuToCpu, endDataFromGpuToCpu) * 1e-3; // definerer en variabel for nedlastings tid
    std::cout << "Time spent sending data to CPU: " << timeForDataFromGpuToCpu << " seconds\n" << std::endl; // skriver ut tiden for nedlasting til CPU fra GPU
    std::cout << "Total time used by GPU: " << timeForGpuUploadEdge + timeForGpuEdge + timeForDataFromGpuToCpu << " seconds\n" << std::endl; // den totale tiden for hele prsesseringen

    
    cv::imshow("Original Image", image1); // bildet som ble brukt orginalt
    cv::imshow("GPU-Processed Image (Edge Detection)", resultGpuEdge); // resultat bildet etter kantdeteksjon blir vist
    cv::waitKey(0); 
    cv::destroyAllWindows(); 

}
