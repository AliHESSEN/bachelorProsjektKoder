#include <iostream>
#include "deteksjonGPU.h"


/**
 * @brief Performs object detection in a video using GPU-based background subtraction.
 *
 * This function utilizes a GPU-based BackgroundSubtractorMOG2 to detect moving objects in each frame of a video.
 * Detected objects are highlighted with bounding rectangles.
 *
 * @note The current implementation detects objects based on contours but does not recognize the detected objects.
 *
 * @details
 * The function starts by opening the specified video file for detection. It creates a CPU-based background subtractor
 * object and a GPU-based background subtractor object using BackgroundSubtractorMOG2. Then, it enters an infinite loop
 * to process each frame of the video.
 *
 * For each frame, the function measures the total time taken for processing one frame and the time taken for transferring
 * data to the GPU. It reads a frame from the video and converts it to GPU format. Background subtraction is then applied
 * on the GPU to obtain a mask. This mask is downloaded from the GPU to the CPU, and thresholding is performed to obtain
 * a binary mask. Contours are found in the binary mask, and objects with contour areas larger than 100 are considered
 * detections. Bounding rectangles are drawn around the detected objects.
 *
 * The original frame along with the processed mask is displayed. The function waits for a key press with a timeout of
 * 30 milliseconds. If the user presses the 'Esc' key, the loop breaks, and the video capture object is released, closing
 * all OpenCV windows.
 *
 * Execution time metrics are also printed, including data transfer time to the GPU, processing time on the GPU, and
 * total execution time.
 */

void detectObjectsWithGPU()
{
    
    cv::VideoCapture videoCapture("C:/Users/lostc/Desktop/highway.mp4"); // videofil for deteksjon
     
    
    cv::Ptr<cv::BackgroundSubtractorMOG2> bakgrunnsdetektorCPU = cv::createBackgroundSubtractorMOG2(100, 40); // lager bakgrunnsdetektorobjekt som skal brukes til å skille objekter fra bakgrunnen, dette er oprettet på CPU

    // bakgrunnsdetektorobjekt på GPU 
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> bakgrunnsdetektorGPU = cv::cuda::createBackgroundSubtractorMOG2(100, 40);

    
    while (true) 
    {
        
        double startTidTotal = static_cast<double>(cv::getTickCount()); // starter målingen av tid

        
        double startTidGPU = static_cast<double>(cv::getTickCount()); // begynner måling av tiden for sende data til GPU

        // for å lese frames
        cv::Mat ramme;
        bool rammeLest = videoCapture.read(ramme);
        if (!rammeLest)
            break;

        // Konverterer rammen til GPU-format
        cv::cuda::GpuMat gpuRamme(ramme);

        
        double sluttTidGPU = static_cast<double>(cv::getTickCount()); // stopper måling av tiden for å sende data til GPU
        double gpuOverføringsTid = (sluttTidGPU - startTidGPU) / cv::getTickFrequency(); // variabel for overføring og regner ut tiden
        std::cout << "Dataoverføringstid til GPU: " << gpuOverføringsTid << " sekunder" << std::endl;

        
        double startTidProsessering = static_cast<double>(cv::getTickCount()); // Starter måling av tiden for GPU-prosessering

        
        cv::cuda::GpuMat gpuMaske;
        bakgrunnsdetektorGPU->apply(gpuRamme, gpuMaske); // gjør bakgrunnsdeteksjon på GPU

        
        cv::Mat maske;
        gpuMaske.download(maske); // Laster ned masken fra GPU til CPU

        // threshold for masken for å få binært result
        cv::threshold(maske, maske, 254, 255, cv::THRESH_BINARY);

        // Finn konturer i den binære masken
        std::vector<std::vector<cv::Point>> konturer;
        cv::findContours(maske, konturer, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        std::vector<cv::Rect> deteksjoner;
        for (const auto& kontur : konturer) 
        {
            double arealAvKonturer = cv::contourArea(kontur);
            if (arealAvKonturer > 100)  // ser  om området av konturen er større enn 100
            {
                cv::Rect boks = cv::boundingRect(kontur); // finner rektanglet
                deteksjoner.push_back(boks); // legger objektet til i deteksjonlisten
            }
        }

        
        for (const auto& deteksjon : deteksjoner) 
        {
            cv::rectangle(ramme, deteksjon, cv::Scalar(0, 255, 0), 3); // Tegner rektangler rundt detekterte objekter
        }

        
        cv::imshow("Ramme", ramme); // viser video
        cv::imshow("Maske", maske); // viser masken

       
        int tast = cv::waitKey(30);
        if (tast == 27)
            break;

        
        double sluttTidProsessering = static_cast<double>(cv::getTickCount()); // variabel som blir brukt for måling av tiden brukt for GPU-prosessering
        double prosesseringsTidGPU = (sluttTidProsessering - startTidProsessering) / cv::getTickFrequency();
        std::cout << "Prosesseringstid (GPU): " << prosesseringsTidGPU << " sekunder" << std::endl;

        
        double sluttTidTotal = static_cast<double>(cv::getTickCount()); // variabel for måling av totaltid tatt for hele prosessen
        double totalKjøretidGPU = (sluttTidTotal - startTidTotal) / cv::getTickFrequency();
        std::cout << "Total kjøretid (GPU): " << totalKjøretidGPU << " sekunder" << std::endl;
    }

    
    videoCapture.release();
    cv::destroyAllWindows();
}
