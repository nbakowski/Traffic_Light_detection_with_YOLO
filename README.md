# Traffic light detection program using YOLO26
The model was trained on the COCO128 dataset. The program allows the detection of traffic light using a YOLO model.
## Modes
There are two modes of operation. The first one uses a YOLO model to determine the position of a traffic light object, but it does not return the color value. That is handled by HSV analysis of the area where traffic light was detected.
Another mode is using a model that has specified classes coressponding to colors. It does not involve HSV detection. 
Tou can choose to use both modes to analyze a webcam feed or video files from a dedicated folder.
## Changing settings
Most setting are located in the **setting.py** file. You can change the model and mode in which the program functions according to your needs.

## Examples
<img width="2559" height="1439" alt="1_wet" src="https://github.com/user-attachments/assets/f13a85ab-f60d-4300-8154-4c6a7b467e35" />
<img width="2559" height="1439" alt="2_wet" src="https://github.com/user-attachments/assets/1a8218de-1e1e-458a-9806-ed3d718bf22e" />

## Resources
The COCO dataset: https://cocodataset.org/#home \
Pretrained models: https://docs.ultralytics.com/models/

## License
MIT License

## Acknowledgments
- COCO dataset creators and maintainers
- Ultralytics team for YOLO implementation
- Contributors and testers

## Contact
n.bakowski@icloud.com

---

**Note**: This project is for educational and research purposes. Ensure compliance with local regulations when deploying traffic monitoring systems.
