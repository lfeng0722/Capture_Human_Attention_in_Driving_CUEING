This is the repo of the work [CUEING A pioneer work of encoding human gaze for autonomous driving].

## Overview
All code is based on Pytorch and some very common libraries.
    
    pip install requirements.txt

Please down human driving gaze datasets from their repos first, for example BDD-A can be downloaded from  https://bdd-data.berkeley.edu/. If they provide video in their repo, please process their video into images following their steps.


And download the YoloV5 from  https://github.com/ultralytics/yolov5 (Release 5.0), then by using our pre-trained YoloV5 Weight 'yolov5l.pt' and camera image from the dataset,  you can gain the bounding box information of your specific dataset.

Then, Compute the Grid for your dataset (training, validation and testing sets) 
    
    run compute_grid.sh

Now, we can move to cleansing method
### Cleansing Method
Input image cleansing:

    run Input_cleansing.sh

Ground truth cleansing:

    run gt_cleansing.sh


### Training and Testing
Training:
    
    1. Compute the Grid for cleansed training set.
    2. run train.sh
    3. the weight of the model will be saved as 'weight.pt'

Testing:
    
    1. Compute the Grid for cleansed testing set.
    2. run test.sh





