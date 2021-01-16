# Machine_Teachers

Algo bot using 3 different signals.

-LSTM model with a 5% difference between real and predicted price to search for long entry.

- 12 over 26 ema crossover for a buy signal after the 5% difference. 


- A "buy" signal on the VADER sentiment analysis indicating that we should stay long after getting the initial entry off of the LSTM buy signal and then the classic 12 over 26 EMA crossover buy signal.

