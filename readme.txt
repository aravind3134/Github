To execute the code: 
1. Please go to the Github/code folder and run Nascent_2.py file.
2. Inputs to this file are: paths to training and testing images of mountain and road bikes. Please ensure the correct way of entering the path WITH NO Quotes and WITH '/' symbol at the end. The example input is shown when the code is run.
3. The code takes care of randomly selecting random number of images (10-20), from training data set and moves it to the testing folder.
4. Ensure that testing folder for both Mountain bike and Road bike is created and is empty. Even if some of the test images are present in the test folder, the code moves it back to the training set and randomly chooses the images, thus decreasing the chance of overfit on training on certain train images.
5. The code is executed 5-fold, to ensure that the code performs well across multiple executions and the average value is taken as final accuracy.
6. The report is included in the report folder.

To be included in the repo:
1. The train.py and test.py split is to be done. 
2. VIsualization in tensorboard is to be done and screenshot is to be attached.
