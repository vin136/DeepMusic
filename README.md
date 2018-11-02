# DeepMusic
Here's a LSTM generating some convincing music


![](https://github.com/vin136/DeepMusic/blob/master/myrnn.png)

Multi-layer recurrent neural networks for training and sampling from texts, inspired by [karpathy/char-rnn](https://github.com/karpathy/char-rnn).

### Requirements

This code is written in Python 3 and it requires the [Keras](https://keras.io) deep learning library.

### Usage

All input data should be placed in the `data/` directory. The example `input.txt` is taken from the [Nottingham Dataset (Cleaned)](https://github.com/jukedeck/nottingham-dataset).

To train the model with default settings:
```bash
$ python train.py
```

To sample the model:
```bash
$ python sample.py 100
```

The sampled outputs can be seen in the `outputs` folder.The outputs are sampled from two models .The model consists of three layer lstm with recurrent dropout.You can look at the output file trained without reccurent dropout(plain dropout) `orginal100.txt`.To listen to the output live paste any one of the outputs [here](https://abcjs.net/abcjs-editor.html)...enjoy....)



