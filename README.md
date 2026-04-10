# Echo State Network with NARMA performance task
Recurrent neural network model with a fading memory for nonlinear information processing.
## Reservoir dynamics
The activation function of the Echo State Network and the reservoir hyperparameters are taken from the article
[High-performance reservoir computing with fluctuations in linear networks](https://drive.google.com/file/d/1Pihq4hKxPAcQSGHvPre05wor9sDo88Ob/view). 
The model is able to perform ensemble learning with spatial multiplexing.
The repository includes a performance task NARMA in `NARMA.py` and some tools to run the code in `tools.py`.
## Machine learning results
`main.ipynb` presents example code for running and evaluating the ESN. A feedback gain of 1.0 was used for a big single component reservoir
and 0.9 for spatially multiplexed systems. An input gain of 0.05 was used for all simulations. The results can be found from `figures/ESN_NARMA_results.png`.
