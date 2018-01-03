#include <vector>
#include <tuple>
#include <map>
#include <tgmath.h>

class Layer;
class Neuron;
class Filter;
class NetMath;
class NetUtil;

class Network {
public:
    static std::vector<Network*> netInstances;
    int instanceIndex;
    int iterations;
    int miniBatchSize;
    int channels;
    float learningRate;
    float rmsDecay;
    float rho;
    float lreluSlope;
    float rreluSlope;
    float eluAlpha;
    bool isTraining;
    float dropout;
    double l2;
    double l2Error;
    double l1;
    double l1Error;
    float maxNorm;
    double maxNormTotal;
    double error;
    std::vector<Layer*> layers;
    std::vector<std::tuple<std::vector<double>, std::vector<double> > > trainingData;
    std::vector<std::tuple<std::vector<double>, std::vector<double> > > testData;
    std::map<std::string, float> weightsConfig;
    double (*activation)(double, bool, Neuron*);
    double (*costFunction)(std::vector<double> calculated, std::vector<double> desired);
    std::vector<double> (*weightInitFn)(int netInstance, int layerIndex, int size);

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

    void train (int iterations, int startIndex);

    double test (int iterations, int startIndex);

    void resetDeltaWeights (void);

    void applyDeltaWeights (void);

};


class Layer {
public:
    int netInstance;
    std::string type;
    int size;
    int fanIn;
    int fanOut;
    int channels;
    int filterSize;
    int stride;
    int zeroPadding;
    int inMapValuesCount;
    int inZPMapValuesCount;
    int outMapSize;
    int prevLayerOutWidth;
    bool hasActivation;
    std::vector<Neuron*> neurons;
    std::vector<Filter*> filters;
    std::vector<std::vector<std::vector<std::vector<int> > > > indeces;
    std::vector<std::vector<std::vector<double> > > errors;
    std::vector<std::vector<std::vector<double> > > activations;
    Layer* nextLayer;
    Layer* prevLayer;
    double (*activation)(double, bool, Neuron*);
    double (*activationC)(double, bool, Filter*);
    double (*activationP)(double, bool, Network*);

    Layer (int netI, int s) {};

    virtual ~Layer(void) {} ;

    virtual void assignNext (Layer* l) = 0;

    virtual void assignPrev (Layer* l) = 0;

    virtual void init (int layerIndex) = 0;

    virtual void forward (void) = 0;

    virtual void backward (std::vector<double> expected) = 0;

    virtual void applyDeltaWeights (void) = 0;

    virtual void resetDeltaWeights (void) = 0;

};

class FCLayer : public Layer {
public:

    FCLayer (int netI, int s);

    ~FCLayer (void);

    void assignNext (Layer* l);

    void assignPrev (Layer* l);

    void init (int layerIndex);

    void forward (void);

    void backward (std::vector<double> errors);

    void applyDeltaWeights (void);

    void resetDeltaWeights (void);
};

class ConvLayer : public Layer {
public:

    ConvLayer (int netI, int s);

    ~ConvLayer (void);

    void assignNext (Layer* l);

    void assignPrev (Layer* l);

    void init (int layerIndex);

    void forward (void);

    void backward (std::vector<double> expected) {
        backward();
    }

    void backward (void);

    void applyDeltaWeights (void);

    void resetDeltaWeights (void);

};

class PoolLayer : public Layer {
public:

    PoolLayer (int netI, int s);

    ~PoolLayer (void);

    void assignNext (Layer* l);

    void assignPrev (Layer* l);

    void init (int layerIndex);

    void forward (void);

    void backward (std::vector<double> expected) {
        backward();
    }

    void backward (void);

    void applyDeltaWeights (void) {};

    void resetDeltaWeights (void) {};
};


class Neuron {
    public:
        std::vector<double> weights;
        std::vector<double> deltaWeights;
        std::vector<double> weightGain;
        std::vector<double> weightsCache;
        std::vector<double> adadeltaCache;
        double lreluSlope;
        double rreluSlope;
        double bias;
        double deltaBias;
        double derivative;
        double activation = 0;
        double sum;
        double error;
        double eluAlpha;
        double biasGain;
        double adadeltaBiasCache;
        double biasCache;
        double m;
        double v;
        bool dropped;

        Neuron(void) {}

        void init (int netInstance);
};

class Filter {
public:
    std::vector<std::vector<std::vector<double> > > weights;
    std::vector<std::vector<std::vector<double> > > deltaWeights;
    std::vector<std::vector<std::vector<double> > > weightGain;
    std::vector<std::vector<std::vector<double> > > weightsCache;
    std::vector<std::vector<std::vector<double> > > adadeltaCache;
    std::vector<std::vector<double> > activationMap;
    std::vector<std::vector<double> > sumMap;
    std::vector<std::vector<double> > errorMap;
    std::vector<std::vector<bool> > dropoutMap;
    double lreluSlope;
    double rreluSlope;
    double bias;
    double deltaBias;
    double derivative;
    double activation;
    double sum;
    double error;
    double eluAlpha;
    double biasGain;
    double adadeltaBiasCache;
    double biasCache;
    double m;
    double v;
    bool dropped;

    Filter (void) {}

    void init (int netInstance);
};


class NetMath {
public:
    template <class T>
    static double sigmoid(double value, bool prime, T* neuron);

    template <class T>
    static double tanh(double value, bool prime, T* neuron);

    template <class T>
    static double lecuntanh(double value, bool prime, T* neuron);

    template <class T>
    static double relu(double value, bool prime, T* neuron);

    template <class T>
    static double lrelu(double value, bool prime, T* neuron);

    template <class T>
    static double rrelu(double value, bool prime, T* neuron);

    template <class T>
    static double elu(double value, bool prime, T* neuron);

    static double meansquarederror (std::vector<double> calculated, std::vector<double> desired);

    static double crossentropy (std::vector<double> target, std::vector<double> output);

    static double vanillaupdatefn (int netInstance, double value, double deltaValue);

    static double gain(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex);

    static double gain(int netInstance, double value, double deltaValue, Filter* filter, int c, int r, int v);

    static double adagrad(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex);

    static double adagrad(int netInstance, double value, double deltaValue, Filter* filter, int c, int r, int v);

    static double rmsprop(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex);

    static double rmsprop(int netInstance, double value, double deltaValue, Filter* filter, int c, int r, int v);

    static double adam(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex);

    static double adam(int netInstance, double value, double deltaValue, Filter* filter, int c, int r, int v);

    static double adadelta(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex);

    static double adadelta(int netInstance, double value, double deltaValue, Filter* filter, int c, int r, int v);

    static std::vector<double> uniform (int netInstance, int layerIndex, int size);

    static std::vector<double> gaussian (int netInstance, int layerIndex, int size);

    static std::vector<double> lecununiform (int netInstance, int layerIndex, int size);

    static std::vector<double> lecunnormal (int netInstance, int layerIndex, int size);

    static std::vector<double> xavieruniform (int netInstance, int layerIndex, int size);

    static std::vector<double> xaviernormal (int netInstance, int layerIndex, int size);

    static std::vector<double> softmax (std::vector<double> values);

    static void maxPool (PoolLayer* layer, int channels);

    static void maxNorm(int netInstance);

    static double sech (double value);
};

class NetUtil {
public:

    static void shuffle (std::vector<std::tuple<std::vector<double>, std::vector<double> > > &values);

    static std::vector<std::vector<double> > addZeroPadding (std::vector<std::vector<double> > map, int zP);

    static std::vector<std::vector<double> > convolve(std::vector<double> input, int zP,
        std::vector<std::vector<std::vector<double> > > weights, int channels, int stride, double bias);

    static std::vector<std::vector<double> > arrayToMap (std::vector<double> array, int size);

    static std::vector<std::vector<std::vector<double> > > arrayToVolume (std::vector<double> array, int channels);

    template <class T>
    static std::vector<std::vector<std::vector<T> > > createVolume (int depth, int rows, int columns, T value);

    static std::vector<std::vector<double> > buildConvErrorMap (int paddedLength, Layer* nextLayer, int filterI);

    static void buildConvDWeights (ConvLayer* layer);

    static std::vector<double> getActivations (Layer* layer);

    static std::vector<double> getActivations (Layer* layer, int mapStartI, int mapSize);

};


// For easier debugging
// void printv(std::vector<double> values) {

//     EM_ASM(window.printfVector = []);

//     for (int i=0; i<values.size(); i++) {
//         EM_ASM_({
//             window.printfVector.push($0)
//         }, values[i]);
//     }

//     EM_ASM(console.log(window.printfVector));
// }

// void printv(std::vector<std::vector<double>> values) {
//     EM_ASM(window.printfVector = []);

//     for (int i=0; i<values.size(); i++) {

//         EM_ASM_({window.printfVector[$0] = []}, i);

//         for (int j=0; j<values[i].size(); j++) {
//             EM_ASM_({
//                 window.printfVector[$0].push($1)
//             }, i, values[i][j]);
//         }
//     }

//     EM_ASM(console.log(window.printfVector));
// }


// void printv(std::vector<std::vector<std::vector<double>>> values) {
//     EM_ASM(window.printfVector = []);

//     for (int i=0; i<values.size(); i++) {

//         EM_ASM_({window.printfVector[$0] = []}, i);

//         for (int j=0; j<values[i].size(); j++) {

//             EM_ASM_({window.printfVector[$0][$1] = []}, i, j);

//             for (int k=0; k<values[i][j].size(); k++) {
//                 EM_ASM_({
//                     window.printfVector[$0][$1].push($2)
//                 }, i, j, values[i][j][k]);
//             }
//         }
//     }

//     EM_ASM(console.log(window.printfVector));
// }