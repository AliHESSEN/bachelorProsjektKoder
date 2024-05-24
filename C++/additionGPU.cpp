#include <iostream>
#include "additionGPU.h"


/**
 * @brief Performs addition operation on four images using GPU.
 *
 * This function takes four input images and performs addition operation on them
 * using GPU. It measures the time taken for the addition operation
 * and displays the original images along with the processed results.
 *
 * @param image1 The first input image.
 * @param image2 The second input image.
 * @param image3 The third input image.
 * @param image4 The fourth input image.
 *
 * @details
 * The function follows these steps:
 * - Uploads the input images to GPU memory.
 * - Performs addition operation on GPU using OpenCV's cuda::addWeighted function.
 *   - The images are added in sequence with specific weights.
 *   - The weights are chosen to ensure that each image contributes equally to the final result.
 * - Measures the time taken for GPU addition.
 * - Downloads the result back to CPU memory.
 * - Displays the original images and the processed results using OpenCV's imshow function.
 * - Destroys all OpenCV windows before the next iteration.
 *
 * @param image1 The first input image for addition.
 * @param image2 The second input image for addition.
 * @param image3 The third input image for addition.
 * @param image4 The fourth input image for addition.
 */


void additionAlgoGPU(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& image3, const cv::Mat& image4)
{
    // GPU prosessering
    cv::cuda::GpuMat gpuImage1, gpuImage2, gpuImage3, gpuImage4, gpuResultAddition; // definere GPU matriser for bildene

    cv::cuda::Event startGpuUploadAddition, endGpuUploadAddition; // cuda events som skal brukes til å måle opplastning til GPU
    startGpuUploadAddition.record(); // starter måling av tid for lasting opp av bilder til GPU 
    // laster opp bilder til GPU
    gpuImage1.upload(image1); 
    gpuImage2.upload(image2); 
    gpuImage3.upload(image3); 
    gpuImage4.upload(image4); 
    endGpuUploadAddition.record(); // Stopper opptak av tid for opplasting til GPU

    cv::cuda::Event startGpuAddition, endGpuAddition; // events som skal brukes for selve addisjonen

    try
    {
        // kjører addisjon på GPU v
        startGpuAddition.record(); // starter måling av tid
        cv::cuda::addWeighted(gpuImage1, 0.25, gpuImage2, 0.25, 0.0, gpuResultAddition); // legger sammen gpuImage1 og gpuImage2 med vektene 0.25
        cv::cuda::addWeighted(gpuResultAddition, 0.5, gpuImage3, 0.5, 0.0, gpuResultAddition); // legger sammen forrige resultat med gpuImage3 med vektene 0.5
        cv::cuda::addWeighted(gpuResultAddition, 0.75, gpuImage4, 0.75, 0.0, gpuResultAddition); // legger sammen forrige resultat med gpuImage4 med vektene 0.75
        endGpuAddition.record(); // stopper tiden
        endGpuAddition.waitForCompletion(); // Venter på at GPU addisjonen skal bli ferdig
    }
    catch (cv::Exception& erroAddGPU)
    {
        std::cerr << "OpenCV exception (GPU): " << erroAddGPU.what() << std::endl; // bruker dette for å få mer infor om feil på GPU
        return; 
    }

    
    float timeForGpuUpload = cv::cuda::Event::elapsedTime(startGpuUploadAddition, endGpuUploadAddition) * 1e-3; // regn tid for opplasting til GPU
    float timeForGpuAdd = cv::cuda::Event::elapsedTime(startGpuAddition, endGpuAddition) * 1e-3; // regner ut tiden for addisjon og konverterer til sekunder ved 1e-3
    std::cout << "----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\nTime spent sending data to GPU: " << timeForGpuUpload << " seconds" << std::endl; 
    std::cout << "Time spent by GPU for addition: " << timeForGpuAdd << " seconds" << std::endl; 

    // Konverter tilbake til CPU matrise
    cv::Mat resultGpuAddition; // definerer en CPU matrisen som skal brukes for resultatet
    cv::cuda::Event startGpuDownload, endGpuDownload; // lager nye event som skal brukes til å måle download tid til CPU 
    startGpuDownload.record(); // starter måling av tid for download fra CPU til GPU 
    gpuResultAddition.download(resultGpuAddition); // laster ned resultatet fra GPU til CPU
    endGpuDownload.record(); // stopper målingen av tid
    endGpuDownload.waitForCompletion(); // inkludert dette for å være sikker på at downloaded er feridg

    // regner ut og skriver ut tidene
    float timeFromGpuToCpu = cv::cuda::Event::elapsedTime(startGpuDownload, endGpuDownload) * 1e-3; // brukes fo tid for nedlasting fra GPU
    std::cout << "Time spent sending data from GPU to CPU: " << timeFromGpuToCpu << " seconds\n" << std::endl; 
    std::cout << "Total time used by GPU: " << timeForGpuUpload + timeForGpuAdd + timeFromGpuToCpu << " seconds\n" << std::endl; // skriver ut total tid brukt av GPU

    
    cv::imshow("Original Image 1", image1); 
    cv::imshow("Original Image 2", image2); 
    cv::imshow("Original Image 3", image3); 
    cv::imshow("Original Image 4", image4); 
    cv::imshow("CUDA-Processed Image (Addition)", resultGpuAddition); // viser resultatet etter addisjon
    cv::waitKey(0); 
    cv::destroyAllWindows(); 
}
