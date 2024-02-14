from ultralytics import YOLO



"""
NOTE: if you converted to trt with ultralytics you will get an error while engine serialization.
to fix comment out the metadata output in file ultralytics/engine/exporter.py.
        
see: https://github.com/ultralytics/ultralytics/issues/4597
"""

# for details https://docs.ultralytics.com/modes/export/

model = YOLO('path/to/your_model.pt')

model.export(format='engine',device=0,imgsz=(384,640),workspace=8)


