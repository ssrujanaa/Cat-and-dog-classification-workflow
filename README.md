# Cat-and-dog-classification-workflow
(Binary Classification with VGG16 workflow using Keras & Optuna (Otpkeras))

Workflow to classify whether images contain either a dog or a cat using the kaggle dogs vs cats dataset.(https://www.kaggle.com/c/dogs-vs-cats/data). 

## Running the Workflow

* Clone the respository using the command `git clone <repository link>`
* `cd` into the `lung-instance-segmentation-workflow` directory
*  [Optional] If you want to add your own docker image, go to `workflow.py` file and change the image in 

    ```python
    unet_wf_cont = Container(
                "unet_wf",
                Container.DOCKER,
                image="docker://vedularaghu/unet_wf:latest"
            )
    ``` 
    
    part, to the link to your docker image
* Run the workflow script using the command `python3 workflow.py`
* Check the predicted masks, model.h5 file, and Checkpoint file in `wf-output` folder

## Executing Standalone Scripts

* Clone the respository using the command `git clone <repository link>`
* `cd` into the `lung-instance-segmentation-workflow/bin` directory
* Use the command `pip3 -r requirements.txt` to install the required packages
* Use the command `python3 <filename>` command to run the `preprocess.py`, `train_model.py`, and `prediction.py` files
