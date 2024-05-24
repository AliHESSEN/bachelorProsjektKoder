#include <iostream>
#include "deteksjonCPU.h"



/**
 * @brief Detects moving objects in a video using CPU-based background subtraction.
 *
 * This function uses a BackgroundSubtractorMOG2 to detect moving objects in a specified region of interest (ROI) in each frame of a video.
 * The detected objects are highlighted with bounding boxes.
 *
 * @note The current implementation detects objects based on contours but does not recognize the detected objects.
 */

void deteksjonAlgoCPU()
{
    
    cv::VideoCapture vidCapture("C:/Users/lostc/Desktop/highway.mp4"); // videofilen som skal brukes for deteksjon

       /**
     * @brief Creates a MOG2 background subtractor.
     *
     * This background subtractor is used to separate moving objects from the background in the video.
     *
     */

    // Oppretter en MOG2 bakgrunnsdetektor 
    cv::Ptr<cv::BackgroundSubtractorMOG2> objectDetector = cv::createBackgroundSubtractorMOG2(100, 40); // skal brukes til å skille bevegelige objekter fra bakgrunnen i videoen

    while (true)
    {
        
        double startTime = static_cast<double>(cv::getTickCount()); // starter måling av tid

        cv::Mat frame;
        
        bool ifFrameHaveBeenRead = vidCapture.read(frame); // leser en ramme fra videoen
      
        if (!ifFrameHaveBeenRead)   // stopper loopen hvis ingenting ble ''lest'' , dvs rammer
            break;

        // Definerer regionen av interesse (ROI) i rammen
        cv::Mat regionOfInterest = frame(cv::Rect(500, 340, 300, 380));

        

        cv::Mat mask;
        objectDetector->apply(regionOfInterest, mask); // bruker bakgrunnsdetektoren på ROI for å skille objektene fra bakgrunnen
       
        cv::threshold(mask, mask, 254, 255, cv::THRESH_BINARY); // threshold verdier 

        
        std::vector<std::vector<cv::Point>> contours;
        
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE); // Finn konturer i masken fra ChatGPT
        std::vector<cv::Rect> detections;
        
        for (const auto& cnt : contours) // looper gjennom alle konturene
        {
           
            double contoursArea = cv::contourArea(cnt);  // regner arealet til konturen
            
            if (contoursArea > 100)  // fjerner områder som er mindre enn 100 for å redusere støy og fjerne objekter som ikke er av interesse
            {
                // gjort med hjelp av ChatGPT
                cv::Rect boundingBoxForObjects = cv::boundingRect(cnt); // Finner rektangelet som rundt objektet
                
                detections.push_back(boundingBoxForObjects); // legger det til i deteksjonlisten
            }
        }

        
        for (const auto& boundingBox : detections) // tegner rektangler rundt detekterte objekter
        {
            cv::rectangle(regionOfInterest, boundingBox, cv::Scalar(0, 255, 0), 3);
        }

        
        cv::imshow("Frame", frame); // viser videoen
        cv::imshow("roi", regionOfInterest); // viser region of intereset, dvs det lille området av interesse
        cv::imshow("Mask", mask); // viser masken

        
        int key = cv::waitKey(30); // stopper hvis man tryker ''esc''
        if (key == 27)
            break;

        // stopper måling av tiden
        double endTime = static_cast<double>(cv::getTickCount());
       
        double detectionTime = (endTime - startTime) / cv::getTickFrequency();  // regner deteksjonstiden
        
        std::cout << "Detection time: " << detectionTime << " seconds" << std::endl;
    }

    
    vidCapture.release();
    
    cv::destroyAllWindows();
}
