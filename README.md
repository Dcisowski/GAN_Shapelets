# GAN_Shapelets

This repository contains Conditional WGAN-GP architecture used for time-series data generation.
It is also using libraries for Shapelets generation in order to compare it with GAN network results.

You can find results from the experiment in **results** folder.
In **checkpoints** directory you can find saved model from different moments of training.
In **training_samples** folder you can find plots of loss functions from training process.

In order to run the experiment you need to execute **run.py** script, **args.yaml** contains configurations 
for training process. You can also specify in there if you want to load saved model or train new one.

There are still some place for improvement (Adding additional Datasets, Enabling users to choose Dataset from the .yaml file)