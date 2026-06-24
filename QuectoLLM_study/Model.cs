namespace QuectoLLM_study;

class Model
{
    private readonly int _V;
    private readonly int _d;

    private readonly double[,] _E;    // [V × d]  embedding matrix
    private readonly double[,] _W;    // [d × V]  output weights
    private readonly double[] _b;     // [V]       output bias

    private double[] _e = [];         // cached embedding of last input word
    private double[] _p = [];         // cached softmax probabilities from last Forward
    private int _lastInput;

    private readonly double[,] _dW;
    private readonly double[] _db;
    private readonly double[] _dErow;

    public int ParameterCount => _V * _d + _d * _V + _V;

    public Model(int vocabSize, int embeddingDim, int seed = 42)
    {
        _V = vocabSize;
        _d = embeddingDim;
        _E = new double[_V, _d];
        _W = new double[_d, _V];
        _b = new double[_V];
        _dW = new double[_d, _V];
        _db = new double[_V];
        _dErow = new double[_d];

        var rng = new Random(seed);
        double range = 1.0 / Math.Sqrt(_d);

        for (int i = 0; i < _V; i++)
            for (int k = 0; k < _d; k++)
                _E[i, k] = (rng.NextDouble() * 2 - 1) * range;

        for (int k = 0; k < _d; k++)
            for (int j = 0; j < _V; j++)
                _W[k, j] = (rng.NextDouble() * 2 - 1) * range;
    }

    // Forward pass: embed input word, project to vocab, apply softmax
    public double[] Forward(int inputIndex)
    {
        _lastInput = inputIndex;

        // Step 1: embedding lookup — grab row i from E
        _e = new double[_d];
        for (int k = 0; k < _d; k++)
            _e[k] = _E[inputIndex, k];

        // Step 2: logits z = e @ W + b
        var z = new double[_V];
        for (int j = 0; j < _V; j++)
        {
            z[j] = _b[j];
            for (int k = 0; k < _d; k++)
                z[j] += _e[k] * _W[k, j];
        }

        // Step 3: softmax over logits
        _p = Softmax(z);
        return _p;
    }

    // Backward pass: compute gradients using the combined softmax+cross-entropy gradient
    public void Backward(int targetIndex)
    {
        // dL/dz = p - one_hot(target)  — the clean combined gradient
        var dz = (double[])_p.Clone();
        dz[targetIndex] -= 1.0;

        // dL/de[k] = sum_j W[k,j] * dz[j] — must be computed BEFORE W is updated
        for (int k = 0; k < _d; k++)
        {
            _dErow[k] = 0;
            for (int j = 0; j < _V; j++)
                _dErow[k] += _W[k, j] * dz[j];
        }

        // dL/dW[k,j] = e[k] * dz[j]
        for (int k = 0; k < _d; k++)
            for (int j = 0; j < _V; j++)
                _dW[k, j] = _e[k] * dz[j];

        // dL/db = dz
        for (int j = 0; j < _V; j++)
            _db[j] = dz[j];
    }

    // SGD update: nudge every parameter opposite its gradient
    public void Update(double learningRate)
    {
        for (int k = 0; k < _d; k++)
            _E[_lastInput, k] -= learningRate * _dErow[k];

        for (int k = 0; k < _d; k++)
            for (int j = 0; j < _V; j++)
                _W[k, j] -= learningRate * _dW[k, j];

        for (int j = 0; j < _V; j++)
            _b[j] -= learningRate * _db[j];
    }

    public double Loss(int targetIndex) => -Math.Log(_p[targetIndex] + 1e-10);

    private static double[] Softmax(double[] z)
    {
        double max = z[0];
        for (int i = 1; i < z.Length; i++)
            if (z[i] > max) max = z[i];

        var exp = new double[z.Length];
        double sum = 0;
        for (int i = 0; i < z.Length; i++)
        {
            exp[i] = Math.Exp(z[i] - max);
            sum += exp[i];
        }

        for (int i = 0; i < exp.Length; i++)
            exp[i] /= sum;

        return exp;
    }
}
