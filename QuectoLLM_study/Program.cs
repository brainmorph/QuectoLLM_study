namespace QuectoLLM_study;

class Program
{
    const int EmbeddingDim = 2;
    const int Epochs = 1000;
    const double LearningRate = 0.01;
    const string CorpusPath = "data/corpus.txt";

    static void Main()
    {
        Console.WriteLine("QuectoLLM — Tiny Next-Word Predictor");
        Console.WriteLine("=====================================");

        var tokenizer = new Tokenizer();
        (int Input, int Target)[] pairs;

        try
        {
            pairs = tokenizer.Load(CorpusPath);
        }
        catch (FileNotFoundException)
        {
            Console.WriteLine($"Error: corpus file not found at '{CorpusPath}'");
            Console.WriteLine("Create a plain text file with 20 sentences, one per line.");
            return;
        }

        int V = tokenizer.VocabSize;
        Console.WriteLine($"Vocabulary ({V} words): {string.Join(", ", tokenizer.IndexToWord)}");
        Console.WriteLine($"Training pairs: {pairs.Length}");

        var model = new Model(V, EmbeddingDim);
        Console.WriteLine($"Parameters: {model.ParameterCount}");
        Console.WriteLine();

        Console.WriteLine("Training...");
        var rng = new Random(0);
        int[] order = [.. Enumerable.Range(0, pairs.Length)];

        for (int epoch = 1; epoch <= Epochs; epoch++)
        {
            // Fisher-Yates shuffle
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            double totalLoss = 0;
            foreach (int idx in order)
            {
                var (input, target) = pairs[idx];
                model.Forward(input);
                totalLoss += model.Loss(target);
                model.Backward(target);
                model.Update(LearningRate);
            }

            if (epoch % 100 == 0)
                Console.WriteLine($"Epoch {epoch,4} / {Epochs}   avg loss = {totalLoss / pairs.Length:F4}");
        }

        Console.WriteLine();
        Console.WriteLine("Training complete. Enter a word to get next-word predictions.");
        Console.WriteLine("Press Enter with no input to quit.");
        Console.WriteLine();

        while (true)
        {
            Console.Write("Word: ");
            var input = Console.ReadLine()?.Trim().ToLowerInvariant();
            if (string.IsNullOrEmpty(input)) break;

            if (!tokenizer.WordToIndex.TryGetValue(input, out int idx))
            {
                Console.WriteLine($"  '{input}' is not in the vocabulary.");
                Console.WriteLine($"  Try one of: {string.Join(", ", tokenizer.IndexToWord)}");
                Console.WriteLine();
                continue;
            }

            var probs = model.Forward(idx);

            var top3 = probs
                .Select((p, i) => (Prob: p, Index: i))
                .OrderByDescending(x => x.Prob)
                .Take(3);

            Console.WriteLine("  Predictions:");
            foreach (var (prob, wordIdx) in top3)
                Console.WriteLine($"    {tokenizer.IndexToWord[wordIdx],-15} {prob:P1}");
            Console.WriteLine();
        }
    }
}
