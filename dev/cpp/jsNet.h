#include <vector>
#include <tuple>
#include <tgmath.h>

class Layer;
class Neuron;
class NetMath;
class NetUtil;

class Network {
public:
    static std::vector<Network*> netInstances;
    int iterations;
    float learningRate;
    float rmsDecay;
    float rho;
    std::vector<Layer*> layers;
    std::vector<std::tuple<std::vector<double>, std::vector<double> > > trainingData;
    std::vector<std::tuple<std::vector<double>, std::vector<double> > > testData;
    double (*activation)(double, bool, Neuron*);
    double (*costFunction)(std::vector<double> calculated, std::vector<double> desired);

    int updateFnIndex;

    Network () {}

    ~Network ();

    static int newNetwork(void);

    static void deleteNetwork(void);

    static void deleteNetwork(int index);

    static Network* getInstance(int i);

    void joinLayers();

    std::vector<double> forward (std::vector<double> input);

    void backward (std::vector<double> expected);

    void train (void);

    double test (void);

    void resetDeltaWeights (void);

    void applyDeltaWeights (void);

    Layer* getLayer(int i);
};

class Layer {
public:
    int netInstance;
    int size;
    std::vector<Neuron*> neurons;
    Layer* nextLayer;
    Layer* prevLayer;
    double (*activation)(double, bool, Neuron*);

    Layer(int netI, int s);

    ~Layer ();

    void assignNext (Layer* l);

    void assignPrev (Layer* l);

    void init (int layerIndex);

    void forward (void);

    void backward (std::vector<double> expected);

    void applyDeltaWeights (void);

    void resetDeltaWeights (void);
};

class Neuron {
    public:
        std::vector<double> weights;
        std::vector<double> deltaWeights;
        std::vector<double> weightGain;
        std::vector<double> weightsCache;
        double bias;
        double deltaBias;
        double derivative;
        double activation;
        double sum;
        double error;
        double lreluSlope;
        double rreluSlope;
        double eluAlpha;
        double biasGain;
        double biasCache;
        double m;
        double v;

        Neuron(void) {}

        void init (int netInstance);
};


class NetMath {
public:
    static double sigmoid(double value, bool prime, Neuron* neuron);

    static double meansquarederror (std::vector<double> calculated, std::vector<double> desired);

    static double vanillaupdatefn (int netInstance, double value, double deltaValue);

    static double gain(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex);

    static double adagrad(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex);

    static double rmsprop(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex);

    static double adam(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex);

    static std::vector<double> softmax (std::vector<double> values);
};