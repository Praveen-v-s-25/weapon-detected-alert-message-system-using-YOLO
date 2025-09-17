# weapon-detected-alert-message-system-using-YOLO
when ever the camera detects the Weapon like gun and knife it captures the image and trigger  the whatsApp message with an alert message and image of the weapon detected to the concern office to prevent attacks . 
1.Dataset-The dataset contains train, test and validation with data.yaml file.
2.The data.yaml is important for training the yolo model.(you can also import dataset from the Kaggle or Roboflow)
3.Code folder contains Yolov11 notebook to train and test the model make sure the dataset paths are correct.
4.save the trained model weights best.pt the weights of the trained model
5. deploy the model using the Flask "best.pt" the trained model weights is used to deploy the model.
6. create a index.html website code and make it to present in the templates folder to connect with the flask.
7. Run flask file command Python app.py
8. if the WhatsApp alert message is not working properly check internet connection 0r change Pywhatkit version.
10. Login to WhatsApp web so that alert message will send.
