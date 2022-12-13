# GreatMusicians
Project that generates different styles of music.

## LSTM-based model

To try a basic LSTM-based music generator, navigate to /LSTM/code, and run 
```
python3 main.py
``` 
Then it will reuse the local model to make prediction. You can find a `generated.mid` output in /LSTM/code

## Transformer-based model
To train a Transformer-based music generator from the scratch, simple navigate into /Transformer, and run 

```
python3 train.py
```

We also provide a pre-trained Jazz generator model weights for you to download: https://drive.google.com/file/d/1FJBM3I214jJQqay9lFeClTAMnQvl7Bs9/view?usp=share_link

You can find a `generated-jazz.mid` in /Transformer/output/.
