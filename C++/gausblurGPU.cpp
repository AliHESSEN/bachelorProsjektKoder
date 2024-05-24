#include <iostream>
#include "gausblurGPU.h"


/**
 * @brief Performs Gaussian blur operation on an input image using GPU acceleration.
 *
 * This function takes an input image and performs Gaussian blur operation on it using GPU acceleration.
 * It measures the time taken for the blur operation on GPU and displays the original image along with the
 * processed result. Additionally, it prints out the time spent for data uploading to GPU, processing on GPU,
 * and downloading the result back to CPU.
 *
 * @param image The input image on which Gaussian blur operation is to be performed.
 *
 * @details The function first uploads the input image data to the GPU memory. Then, it creates a Gaussian filter
 * with the specified kernel size and applies it to the input image using OpenCV's GPU-accelerated functions.
 * The resulting blurred image is downloaded back to the CPU memory. The function also measures and prints the
 * time taken for data uploading, processing, and downloading, providing insights into the performance of GPU
 * acceleration for Gaussian blur operation.
 */


void gaussianBlurAlgoGPU(const cv::Mat& image) 
{

    // GPU processing
    cv::cuda::GpuMat gpuImage, gpuResultBlur;

    cv::cuda::Event startGpuUploadBlur, endGpuUploadBlur;
    startGpuUploadBlur.record(); // start recording time
    gpuImage.upload(image);
    endGpuUploadBlur.record(); // stop recording time

    // Create Gaussian filter on GPU
    cv::Ptr<cv::cuda::Filter> gaussianFilter;
   
    gaussianFilter = cv::cuda::createGaussianFilter(gpuImage.type(), gpuImage.type(), cv::Size(31, 31), 0, 0);
    
 

    // Perform Gaussian blur on GPU
    cv::cuda::Event startGpuBlur, endGpuBlur;
    try {
        // Perform Gaussian blur on GPU
        startGpuBlur.record(); // start recording time
        gaussianFilter->apply(gpuImage, gpuResultBlur);
        endGpuBlur.record(); // stop recording time
        endGpuBlur.waitForCompletion();
    }
    catch (cv::Exception& error) {
        std::cerr << "OpenCV exception (GPU blur application): " << error.what() << std::endl;
        return;
    }

    // Calculate and print the time for GPU processing in seconds
    float usedTimeGpuUploadBlur = cv::cuda::Event::elapsedTime(startGpuUploadBlur, endGpuUploadBlur) * 1e-3;
    float usedTimeGpuBlur = cv::cuda::Event::elapsedTime(startGpuBlur, endGpuBlur) * 1e-3;
    std::cout << "----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\nTime spent for uploading to GPU: " << usedTimeGpuUploadBlur << " seconds" << std::endl;
    std::cout << "Time spent by GPU for Gaussian blur: " << usedTimeGpuBlur << " seconds" << std::endl;

    // Convert back to CPU matrix
    cv::Mat resultGpuBlur;
    cv::cuda::Event startGpuDownloadBlur, endGpuDownloadBlur;
    startGpuDownloadBlur.record(); // take time
    gpuResultBlur.download(resultGpuBlur); // send back to CPU
    endGpuDownloadBlur.record(); // stop time
    endGpuDownloadBlur.waitForCompletion();

    // Calculate and print the time for GPU download in seconds
    float usedTimeGpuDownloadBlur = cv::cuda::Event::elapsedTime(startGpuDownloadBlur, endGpuDownloadBlur) * 1e-3;
    std::cout << "Time spent sending data to CPU: " << usedTimeGpuDownloadBlur << " seconds\n" << std::endl;
    std::cout << "Total time spent by GPU: " << usedTimeGpuUploadBlur + usedTimeGpuBlur + usedTimeGpuDownloadBlur << " seconds\n" << std::endl;

    // Display the original image and the result
    cv::imshow("Original Image", image);
    cv::imshow("CUDA-Processed Image (Gaussian Blur)", resultGpuBlur);
    cv::waitKey(0);
    cv::destroyAllWindows(); // Close all OpenCV windows

}