from ultralytics import YOLO


model  = YOLO('weights/best.pt')



results =  model.predict(source="tests/organic.png",conf=0.1,show=True,save=True)