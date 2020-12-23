# AIY Vision Kit - Utils
Libraries and examples for using the AIY Vision Kit with CogniFly (or any RPI Zero actually...)

Before you start, make sure to backup your current RPI Zero W sd card ([instructions here](https://thepihut.com/blogs/raspberry-pi-tutorials/17789160-backing-up-and-restoring-your-raspberry-pis-sd-card)). After that, you can restore the image supplied in [this release](https://github.com/thecognifly/AIYVisionKit_Utils/releases/tag/Alpha). The same release has the dataset and TensorFlow checkpoints used in these step-by-step instructions. All notebooks were created using [Google Colab](https://colab.research.google.com/), so there's no need for a computer with GPU.

1. Start by collecting images for the dataset ([Darth_Vader_Convert_RPI_Videos_2_PNG_images.ipynb](https://github.com/thecognifly/AIYVisionKit_Utils/blob/master/Darth_Vader_Convert_RPI_Videos_2_PNG_images.ipynb))
2. Annotate the images ([Darth_Vader_Annotate_Images.ipynb](https://github.com/thecognifly/AIYVisionKit_Utils/blob/master/Darth_Vader_Annotate_Images.ipynb))
3. Train (actually, it's [trains learning](https://en.wikipedia.org/wiki/Transfer_learning) based on the TensorFlow Object Detection API - [Darth_Vader_Training.ipynb](https://github.com/thecognifly/AIYVisionKit_Utils/blob/master/Darth_Vader_Training.ipynb).
4. Export the model and compile it ([Darth_Vader_Exporting_Testing_Compiling.ipynb](https://github.com/thecognifly/AIYVisionKit_Utils/blob/master/Darth_Vader_Exporting_Testing_Compiling.ipynb))
5. And that's it, you just need to deploy it to AIY Vision Kit (the image supplied already has the model and all the necessary files to test it)!
