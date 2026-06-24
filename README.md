QuectoLLM — Tiny Next-Word Predictor
=====================================
A from-scratch neural network that learns to predict the next word in a sentence.
Built in C# with no third-party libraries. 210 parameters total.


REQUIREMENTS
------------
- .NET 10 SDK
  Download from: https://dotnet.microsoft.com/download


HOW TO RUN
----------
1. Open a terminal.

2. Navigate to the project folder:
      cd QuectoLLM_study/QuectoLLM_study

3. Run the program:
      dotnet run

4. The program will train on the sentences in data/corpus.txt.
   You will see the loss printed every 100 epochs — it should decrease
   over time, which means the model is learning.

5. Once training finishes, type any word from the vocabulary and press Enter.
   The model will print the top 3 predicted next words with probabilities.

6. Press Enter with no input to quit.


CUSTOMISING THE TRAINING DATA
------------------------------
Open QuectoLLM_study/data/corpus.txt and replace the sentences with your own.
Rules:
  - One sentence per line.
  - Plain English words only (punctuation is ignored automatically).
  - Aim for 15–30 sentences with overlapping vocabulary so the model
    has enough repetition to learn from.
  - The vocabulary is capped at 50 unique words. If your corpus has more,
    the least frequent words are dropped.


EXAMPLE SESSION
---------------
  Vocabulary (42 words): the, in, cat, on, ...
  Training pairs: 93
  Parameters: 210

  Training...
  Epoch  100 / 1000   avg loss = 2.2909
  ...
  Epoch 1000 / 1000   avg loss = 1.5121

  Training complete. Enter a word to get next-word predictions.

  Word: cat
    Predictions:
      ate             17.2%
      sat             17.1%
      watched         17.1%

  Word:


WHAT IS "LOSS"?
---------------
Loss measures how surprised the model was by the correct answer.
A loss of 0 would mean it predicted perfectly every time.
As training progresses the loss should decrease, meaning the model is
getting less surprised — i.e. it is learning the patterns in your text.


FILES
-----
  Program.cs      Entry point. Runs the training loop and inference prompt.
  Tokenizer.cs    Reads corpus.txt, builds the vocabulary, creates training pairs.
  Model.cs        The neural network. Forward pass, backprop, and SGD update.
  data/corpus.txt The 20 training sentences. Replace with your own text.
  Plan.txt        Design decisions and architecture notes.
