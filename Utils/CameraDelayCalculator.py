import os
from pathlib import Path
from typing import List
from RMS.Misc import getRmsRootDir

class CameraDelayCalculator:
    def __init__(self, RMS_path: str, delayInterval: int = 10):
        """
        Initialize the Camera Delay Calculator for RMS stations.
        
        Args:
            RMS_path (str): Path to RMS root directory
            delayInterval (int): Time interval between cameras in seconds
        """
        self.rmsPath = Path(RMS_path)
        self.stationsPath = self.rmsPath.parent / "Stations"
        self.delayInterval = delayInterval
        self.cameraList: List[str] = []

    def discoverCameras(self) -> None:
        """
        Walk through the Stations directory structure to discover all camera IDs
        and sort them.
        """
        if not self.stationsPath.exists():
            raise FileNotFoundError(f"Stations path does not exist: {self.stationsPath}")

        # Walk through all directories in the Stations path
        self.cameraList = [
            entry.name
            for entry in os.scandir(self.stationsPath)
            if entry.is_dir()
        ]
        # Sort camera IDs
        self.cameraList.sort()

    def getDelay(self, cameraId: str) -> int:
        """
        Get the calculated delay for a specific camera ID.
        
        Args:
            cameraId (str): The camera ID to look up
            
        Returns:
            int: The calculated delay in seconds
        
        Raises:
            ValueError: If the camera ID is not found in the discovered cameras
        """
        if not self.cameraList:
            self.discoverCameras()
            
        try:
            index = self.cameraList.index(cameraId)
            return index * self.delayInterval
        except ValueError:
            raise ValueError(f"Camera ID not found: {cameraId}")


def main():
    # Example usage with RMS path
    RMS_path = getRmsRootDir()  # You would import this function
    calculator = CameraDelayCalculator(RMS_path)
    
    try:
        # Get delay for a specific camera
        cameraId = "US005D"
        delay = calculator.getDelay(cameraId)
        print(f"Delay for camera {cameraId}: {delay}s")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()