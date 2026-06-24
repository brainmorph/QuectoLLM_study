namespace QuectoLLM_study;

class Tokenizer
{
    public const int MaxVocabSize = 50;

    public string[] IndexToWord { get; private set; } = [];
    public Dictionary<string, int> WordToIndex { get; private set; } = [];
    public int VocabSize => IndexToWord.Length;

    public (int Input, int Target)[] Load(string corpusPath)
    {
        var lines = File.ReadAllLines(corpusPath);
        var sentences = new List<string[]>();
        var freq = new Dictionary<string, int>();

        foreach (var line in lines)
        {
            var words = Tokenize(line);
            if (words.Length < 2) continue;
            sentences.Add(words);
            foreach (var w in words)
                freq[w] = freq.GetValueOrDefault(w) + 1;
        }

        IndexToWord = freq
            .OrderByDescending(kv => kv.Value)
            .Take(MaxVocabSize)
            .Select(kv => kv.Key)
            .ToArray();

        WordToIndex = IndexToWord
            .Select((w, i) => (w, i))
            .ToDictionary(t => t.w, t => t.i);

        var pairs = new List<(int, int)>();
        foreach (var sentence in sentences)
            for (int i = 0; i < sentence.Length - 1; i++)
                if (WordToIndex.TryGetValue(sentence[i], out int a) &&
                    WordToIndex.TryGetValue(sentence[i + 1], out int b))
                    pairs.Add((a, b));

        return pairs.ToArray();
    }

    private static string[] Tokenize(string line)
    {
        var sb = new System.Text.StringBuilder(line.Length);
        foreach (char c in line.ToLowerInvariant())
            sb.Append(char.IsAsciiLetter(c) ? c : ' ');
        return sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }
}
