# LSTM neural networks to forecast spread levels for fixed income. Then these forecasts can be used in an expected return framework (along with treasuries' forecast), to assess the expected return of each asset class.

## Working with 3 different Fixed Income indices, one for Emerging Markets, one for Global Corporate High-Yielders, and one for Investment-Grade, I tried to forecast if their spread levels would remain roughly at their current level or would go up or down considerably.

* My dataset's observations were dates and its dimensions were composed of price changes of different fixed income indices, the SPX, the DXY, commodities, the spread levels of different bond indices, volatility of those indices for which I calculated price changes, short-term US yields levels, and technical indicators like RSI.
* These would be fed into an LSTM neural network, whose architecture can be seen in my files, with a specific timestep to try to predict in a classification problem if spreads would go up (defined as a movement of >=5%), down defined as a movement of <=-5%, or would stay roughly between that interval.
* The accuracy was measured using a confusion matrix, and if the LSTM network was of any use, it'd provide an accuracy for each label above its base rate of 33%. I managed to obtain accuracies that oscillate between 50-70%.
