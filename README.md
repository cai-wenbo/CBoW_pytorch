# Introduction
This is an implementation of the CBoW model, written in pytorch.  

# Way to use
You can run the training_scripts and the inspect_scripts to test the performance of the model.  

The dataset for demo was extracted from BBC news, and the demo embedding model was pretrained on it.   

# To Do
Currently the bottleneck of model training is the generation of the negative targets. Which uses only one cpu, so now whether you train the model on a GPU doesn't make a significant difference on the speed. We can try to use multithread to get boost.   

