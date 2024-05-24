#include <iostream>
#include "histEqGPU.h"

/**
 * @brief Performs histogram equalization on an input image using GPU.
 *
 * This function takes an input image and performs histogram equalization on it using GPU. It measures the time taken for the histogram equalization
 * and displays the original image along with the processed result.
 *
 * @param image1 The input image for histogram equalization.
 *
 * @details
 * The function follows these steps:
 * - Uploads the input image to GPU memory.
 * - Converts the image to grayscale on GPU using OpenCV's cuda::cvtColor function.
 * - Performs histogram equalization on GPU using OpenCV's cuda::equalizeHist function.
 * - Measures the time taken for GPU histogram equalization.
 * - Downloads the result back to CPU memory.
 * - Displays the original image and the processed result using OpenCV's imshow function.
 *
 * @note The input image is expected to be in BGR format.
 *
 * @param image1 The input image for histogram equalization.
 */

void histogramEqualizationAlgoGPU(const cv::Mat& image1)
{
    cv::cuda::GpuMat gpuImage1, gpuResultHistEq; // lgaer to GPU matriser en for input og en for resultat
    cv::cuda::Event startGpuUploadHistEq, endGpuUploadHistEq; // skal bruke disse eventene til å måle opplastings tid til GPU

    startGpuUploadHistEq.record(); // starter målingen
    gpuImage1.upload(image1); // laster opp data til GPU
    endGpuUploadHistEq.record(); // stopper målingen

    cv::cuda::Event startGpuHistEq, endGpuHistEq; // events for som skal brukes til selve histeq operasjonen

    
    startGpuHistEq.record(); // måler tiden
    cv::cuda::cvtColor(gpuImage1, gpuImage1, cv::COLOR_BGR2GRAY); //  gjør om til greyscale
    cv::cuda::equalizeHist(gpuImage1, gpuResultHistEq); // kjøer histeq funksjonen på GPU
    endGpuHistEq.record(); // stopper målingen av tid
    endGpuHistEq.waitForCompletion(); // sørger for at det er feridg før vi fortsetter

    // regner ut tider under her,og konverterer til sekunder
    float timeUsedOnGpuHistEqUpload = cv::cuda::Event::elapsedTime(startGpuUploadHistEq, endGpuHistEq) * 1e-3; // varbel som brukes for opplasting til GPU
    float timeUsedOnGpuHistEq = cv::cuda::Event::elapsedTime(startGpuHistEq, endGpuHistEq) * 1e-3; // varibel for selve operasjonen 
    std::cout << "----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\nTime spent sending data to GPU: " << timeUsedOnGpuHistEqUpload << " seconds" << std::endl; // skriver ut tid for oplastning
    std::cout << "Time spent by GPU for histogram equalization: " << timeUsedOnGpuHistEq << " seconds" << std::endl; // skriver ut tid for operasjonen

    
    cv::Mat resultGpuHistEq; // lager en CPU matris som skal holde resultatet fra GPU
    cv::cuda::Event startGpuHistEqDownload, endGpuHistEqDownload; // lager to nye events for å måle nedlastings tid fra GPU til CPU
    startGpuHistEqDownload.record(); // starter målingen av tid
    gpuResultHistEq.download(resultGpuHistEq); // laster det ned
    endGpuHistEqDownload.record(); // stopper tiden

    float timeUsedOnGpuHistEqDownload = cv::cuda::Event::elapsedTime(startGpuHistEqDownload, endGpuHistEqDownload) * 1e-3; // lager også en varibel for nedlasatings tider
    std::cout << "Time spent sending data from GPU to CPU: " << timeUsedOnGpuHistEqDownload << " seconds \n" << std::endl; 
    std::cout << "Total time used by GPU: " << timeUsedOnGpuHistEqUpload + timeUsedOnGpuHistEq + timeUsedOnGpuHistEqDownload << " seconds\n" << std::endl; // Skriver ut total tid

    
    cv::imshow("Original Image 1", image1); // bildet som ble brukt
    cv::imshow("GPU-Processed Image (Hist. Equalization)", resultGpuHistEq); // resultatet etter operasjonen blir vist
    cv::waitKey(0); 
    cv::destroyAllWindows(); 
}
