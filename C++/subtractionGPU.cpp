#include <iostream>
#include "subtractionGPU.h"

/**
 * @brief Performs GPU-based image subtraction.
 *
 * This function performs image subtraction using the GPU. It takes two input images, converts them to GPU format,
 * subtracts one from the other, and returns the result. It also measures and prints the time taken for data upload
 * to the GPU, subtraction operation on the GPU, and data download from the GPU to the CPU.
 *
 * @param image1 The first input image.
 * @param image2 The second input image.
 *
 * @details
 * The function starts by uploading the input images to GPU memory for processing. It records the time taken for this
 * data upload operation. Then, it performs the subtraction operation using GPU-accelerated functions. The time taken
 * for this operation is also recorded.
 *
 * After the subtraction operation, the result is downloaded from the GPU to CPU memory. The time taken for this data
 * download operation is recorded as well.
 *
 * The function prints out the time taken for each of these operations along with the total time taken for the entire
 * process.
 *
 * Finally, the original input images along with the resulting subtracted image are displayed using OpenCV's imshow
 * function. The function waits for a key press to close the displayed images.
 */


void subtractionAlgoGPU(const cv::Mat& image1, const cv::Mat& image2)
{
    
    cv::cuda::GpuMat gpuImage1, gpuImage2, gpuResultSubtraction; //  GPU matriser variabler for de to bildene og resultatet

    
    cv::cuda::Event startGpuUploadSubtraction, endGpuUploadSubtraction; // event for å laste opp data til GPU
    startGpuUploadSubtraction.record(); // begynner målingen av tid
    gpuImage1.upload(image1); // laster opp bilde 1 til gpu
    gpuImage2.upload(image2); // laster opp bilde 2 til gpu
    endGpuUploadSubtraction.record(); // stopper målingen av tid

    
    cv::cuda::Event startGpuSubtraction, endGpuSubtraction; // events som jeg skal bruke for å måle selve subtraksjonen
    startGpuSubtraction.record(); // starter tid måling
    cv::cuda::subtract(gpuImage1, gpuImage2, gpuResultSubtraction); // gjør subtraksjonen på bilden
    endGpuSubtraction.record(); // stopper tid målingen
    endGpuSubtraction.waitForCompletion(); // bruker dette for å sørge for at det er feridg

    
    float usedTimeGpuUploadSubtraction = cv::cuda::Event::elapsedTime(startGpuUploadSubtraction, endGpuUploadSubtraction) * 1e-3; // opplastingstid variabel konverterer til sekunder 1e-3
    float usedTimeGpuSubtraction = cv::cuda::Event::elapsedTime(startGpuSubtraction, endGpuSubtraction) * 1e-3; // subtraksjonoperasjon variabelen for tiden
    std::cout << "----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\nTid brukt for opplasting til GPU: " << usedTimeGpuUploadSubtraction << " sekunder" << std::endl;
    std::cout << "Tid brukt av GPU for subtraksjon: " << usedTimeGpuSubtraction << " sekunder" << std::endl;

   
    cv::Mat resultGpuSubtraction;  // lager CPU matrise for resultatet
    cv::cuda::Event startGpuDownloadSubtraction, endGpuDownloadSubtraction; // events som bruke for å måle tiden for nedlasting fra GPU til CPU
    startGpuDownloadSubtraction.record(); // stater måling av tiden
    gpuResultSubtraction.download(resultGpuSubtraction); // laster ned resultatet
    endGpuDownloadSubtraction.record(); // stopper målingen av tid
    endGpuDownloadSubtraction.waitForCompletion(); 

    
    float usedTimeGpuDownloadSubtraction = cv::cuda::Event::elapsedTime(startGpuDownloadSubtraction, endGpuDownloadSubtraction) * 1e-3; // variabel for nedlastings tid
    std::cout << "Tid brukt for å sende data til CPU: " << usedTimeGpuDownloadSubtraction << " sekunder\n" << std::endl;
    std::cout << "Total tid brukt av GPU: " << usedTimeGpuUploadSubtraction + usedTimeGpuSubtraction + usedTimeGpuDownloadSubtraction << " sekunder\n" << std::endl;

    
    cv::imshow("Original Image 1", image1); // viser det første originale bildet
    cv::imshow("Original Image 2", image2); // viser det andre originale bildet
    cv::imshow("CUDA-Processed Image (Subtraction)", resultGpuSubtraction); // viser resultatet etter subtraksjonoperasjonen
    cv::waitKey(0); 
    cv::destroyAllWindows(); 
}
