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
    float learningRate;
    std::vector<Layer*> layers;
    std::vector<std::tuple<std::vector<double>, std::vector<double> > > trainingData;
    std::vector<std::tuple<std::vector<double>, std::vector<double> > > testData;
    double (*activation)(double, bool, Neuron*);
    double (*costFunction)(std::vector<double> calculated, std::vector<double> desired);

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
private:
    int netInstance;
public:
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
        double bias;
        double deltaBias;
        double derivative;
        double activation;
        double sum;
        double error;
        double lreluSlope;
        double rreluSlope;
        double eluAlpha;

        Neuron() {}

        void init (void);
};


class NetMath {
public:
    static double sigmoid(double value, bool prime, Neuron* neuron);

    static double meansquarederror (std::vector<double> calculated, std::vector<double> desired);

    static float vanillaupdatefn (int netInstance, float value, float deltaValue);

    static std::vector<double> softmax (std::vector<double> values);
};