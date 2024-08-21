# Receipt OCR


## Requirements:
**Milestone 1:** <br>
Custom OCR Development and Initial Testing 

Receipt Detection 
 
Labelling Dataset whether it is an image or a 
receipt or not 
 
A custom OCR engine capable of extracting text 
from receipts. 
 
Implement pre-processing techniques like image 
enhancement, noise reduction, and skew 
correction. 
 
Initial testing results demonstrate the OCR engine's 
accuracy and reliability on various receipt images.


**Milestone 2:** <br>
LLM Integration and Text Parsing 

Integration of a Large Language Model (LLM) 
trained to recognize and classify receipt elements. 
 
A functional pipeline that processes OCR-extracted 
text through the LLM to produce structured data. 
 
Testing and validation results show the system's 
ability to parse and organize receipt information 
accurately

## Timeline
Task	                                  Days <br>
Dataset prepration	                        3   <br>
Training and Testing 	                    2   <br>
Finetuning and Testing if needed	        2   <br>
LLM Testings 	                            3   <br>
LLM Pipeline setup and formatting	        2   <br>
API deployemnt for cross platform testings	2   <br>
Final pipeline testing and QA	            3   <br>



## How to Run the Project:

1. Install Dependencies:
```shell
pip install -r requirements.txt
```

2. Train YOLOv8 Model:
```shell
python src/yolov8_train.py
```

3. Test Models:
```shell
python src/model_test.py

```

5. Run the Flask API:
```shell
python src/flask_api.py
```
