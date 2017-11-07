#include <stdio.h>
#include <emscripten.h>
#include "Network.cpp"

int main(int argc, char const *argv[]) {
    emscripten_run_script("typeof window!='undefined' && window.dispatchEvent(new CustomEvent('jsNetWASMLoaded'))");
    return 0;
}

extern "C" {

    EMSCRIPTEN_KEEPALIVE
    int newNetwork (void) {
        return Network::newNetwork();
    }

    /* Network config */
    EMSCRIPTEN_KEEPALIVE
    float getLearningRate (int instanceIndex) {
        return Network::getInstance(instanceIndex)->learningRate;
    }

    EMSCRIPTEN_KEEPALIVE
    void setLearningRate (int instanceIndex, float lr) {
        Network::getInstance(instanceIndex)->learningRate = lr;
    }

    EMSCRIPTEN_KEEPALIVE
    double getError (int instanceIndex) {
        return Network::getInstance(instanceIndex)->error;
    }

    EMSCRIPTEN_KEEPALIVE
    void setActivation (int instanceIndex, int activationFnIndex) {
        Network* net = Network::getInstance(instanceIndex);

        switch (activationFnIndex) {
            case 0:
                net->activation = &NetMath::sigmoid<Neuron>;
                break;
            case 1:
                net->activation = &NetMath::tanh<Neuron>;
                break;
            case 2:
                net->activation = &NetMath::lecuntanh<Neuron>;
                break;
            case 3:
                net->activation = &NetMath::relu<Neuron>;
                break;
            case 4:
                net->activation = &NetMath::lrelu<Neuron>;
                break;
            case 5:
                net->activation = &NetMath::rrelu<Neuron>;
                break;
            case 6:
                net->activation = &NetMath::elu<Neuron>;
                break;
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void setCostFunction (int instanceIndex, int fnIndex) {

        Network* net = Network::getInstance(instanceIndex);

        switch (fnIndex) {
            case 0:
                net->costFunction = &NetMath::meansquarederror;
                break;
            case 1:
                net->costFunction = &NetMath::crossentropy;
                break;
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void set_rmsDecay  (int instanceIndex, float rmsDecay) {
        Network::getInstance(instanceIndex)->rmsDecay = rmsDecay;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_rmsDecay (int instanceIndex) {
        return Network::getInstance(instanceIndex)->rmsDecay;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_rho  (int instanceIndex, float rho) {
        Network::getInstance(instanceIndex)->rho = rho;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_rho (int instanceIndex) {
        return Network::getInstance(instanceIndex)->rho;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_lreluSlope  (int instanceIndex, float lreluSlope) {
        Network::getInstance(instanceIndex)->lreluSlope = lreluSlope;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_lreluSlope (int instanceIndex) {
        return Network::getInstance(instanceIndex)->lreluSlope;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_eluAlpha  (int instanceIndex, float eluAlpha) {
        Network::getInstance(instanceIndex)->eluAlpha = eluAlpha;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_eluAlpha (int instanceIndex) {
        return Network::getInstance(instanceIndex)->eluAlpha;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_dropout  (int instanceIndex, float dropout) {
        Network::getInstance(instanceIndex)->dropout = dropout;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_dropout (int instanceIndex) {
        return Network::getInstance(instanceIndex)->dropout;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_l2  (int instanceIndex, float l2) {
        Network::getInstance(instanceIndex)->l2 = l2;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_l2 (int instanceIndex) {
        return Network::getInstance(instanceIndex)->l2;
    }
    EMSCRIPTEN_KEEPALIVE
    void set_l2Error  (int instanceIndex, float l2Error) {
        Network::getInstance(instanceIndex)->l2Error = l2Error;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_l2Error (int instanceIndex) {
        return Network::getInstance(instanceIndex)->l2Error;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_l1  (int instanceIndex, float l1) {
        Network::getInstance(instanceIndex)->l1 = l1;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_l1 (int instanceIndex) {
        return Network::getInstance(instanceIndex)->l1;
    }
    EMSCRIPTEN_KEEPALIVE
    void set_l1Error  (int instanceIndex, float l1Error) {
        Network::getInstance(instanceIndex)->l1Error = l1Error;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_l1Error (int instanceIndex) {
        return Network::getInstance(instanceIndex)->l1Error;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_maxNorm  (int instanceIndex, float maxNorm) {
        Network::getInstance(instanceIndex)->maxNorm = maxNorm;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_maxNorm (int instanceIndex) {
        return Network::getInstance(instanceIndex)->maxNorm;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_maxNormTotal  (int instanceIndex, float maxNormTotal) {
        Network::getInstance(instanceIndex)->maxNormTotal = maxNormTotal;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_maxNormTotal (int instanceIndex) {
        return Network::getInstance(instanceIndex)->maxNormTotal;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_channels  (int instanceIndex, float channels) {
        Network::getInstance(instanceIndex)->channels = channels;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_channels (int instanceIndex) {
        return Network::getInstance(instanceIndex)->channels;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filterSize  (int instanceIndex, float filterSize) {
        Network::getInstance(instanceIndex)->filterSize = filterSize;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_filterSize (int instanceIndex) {
        return Network::getInstance(instanceIndex)->filterSize;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_zeroPadding  (int instanceIndex, float zeroPadding) {
        Network::getInstance(instanceIndex)->zeroPadding = zeroPadding;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_zeroPadding (int instanceIndex) {
        return Network::getInstance(instanceIndex)->zeroPadding;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_stride  (int instanceIndex, float stride) {
        Network::getInstance(instanceIndex)->stride = stride;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_stride (int instanceIndex) {
        return Network::getInstance(instanceIndex)->stride;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_distribution  (int instanceIndex, int distribution) {

        Network* net = Network::getInstance(instanceIndex);
        net->weightsConfig["distribution"] = distribution;

        switch (distribution) {
            case 0:
                net->weightInitFn = &NetMath::uniform;
                break;
            case 1:
                net->weightInitFn = &NetMath::gaussian;
                break;
            case 2:
                net->weightInitFn = &NetMath::xavieruniform;
                break;
            case 3:
                net->weightInitFn = &NetMath::xaviernormal;
                break;
            case 4:
                net->weightInitFn = &NetMath::lecununiform;
                break;
            case 5:
                net->weightInitFn = &NetMath::lecunnormal;
                break;
        }
    }

    EMSCRIPTEN_KEEPALIVE
    int get_distribution (int instanceIndex) {
        return Network::getInstance(instanceIndex)->weightsConfig["distribution"];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_limit  (int instanceIndex, float limit) {
        Network::getInstance(instanceIndex)->weightsConfig["limit"] = limit;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_limit (int instanceIndex) {
        return Network::getInstance(instanceIndex)->weightsConfig["limit"];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_mean  (int instanceIndex, float mean) {
        Network::getInstance(instanceIndex)->weightsConfig["mean"] = mean;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_mean (int instanceIndex) {
        return Network::getInstance(instanceIndex)->weightsConfig["mean"];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_stdDeviation  (int instanceIndex, float stdDeviation) {
        Network::getInstance(instanceIndex)->weightsConfig["stdDeviation"] = stdDeviation;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_stdDeviation (int instanceIndex) {
        return Network::getInstance(instanceIndex)->weightsConfig["stdDeviation"];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_updateFn (int instanceIndex, int fnIndex) {
        Network::getInstance(instanceIndex)->updateFnIndex = fnIndex;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_updateFn (int instanceIndex) {
        return Network::getInstance(instanceIndex)->updateFnIndex;
    }

    EMSCRIPTEN_KEEPALIVE
    void addFCLayer (int instanceIndex, int size) {
        Network::getInstance(instanceIndex)->layers.push_back(new FCLayer(instanceIndex, size));
    }

    EMSCRIPTEN_KEEPALIVE
    void addConvLayer (int instanceIndex, int size) {
        Network::getInstance(instanceIndex)->layers.push_back(new ConvLayer(instanceIndex, size));
    }

    EMSCRIPTEN_KEEPALIVE
    void initLayers (int instanceIndex) {
        Network::getInstance(instanceIndex)->joinLayers();
    }

    EMSCRIPTEN_KEEPALIVE
    double* forward (int instanceIndex, float *buf, int vals) {

        Network* net = Network::getInstance(instanceIndex);
        std::vector<double> input;

        for (int i=0; i<vals; i++) {
            input.push_back((double)buf[i]);
        }

        std::vector<double> activations = net->forward(input);
        std::vector<double> softmax = NetMath::softmax(activations);

        double returnArr[softmax.size()];
        for (int v=0; v<activations.size(); v++) {
            returnArr[v] = softmax[v];
        }

        auto arrayPtr = &returnArr[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    void loadTrainingData (int instanceIndex, float *buf, int total, int size, int dimension) {
        Network* net = Network::getInstance(instanceIndex);
        net->trainingData.clear();

        std::tuple<std::vector<double>, std::vector<double> > epoch;

        // Push training data to memory
        for (int i=0; i<=total; i++) {

            if (i && i%size==0) {
                net->trainingData.push_back(epoch);
                std::get<0>(epoch).clear();
                std::get<1>(epoch).clear();
            }

            if (i%size<dimension) {
                std::get<0>(epoch).push_back((double)buf[i]);
            } else {
                std::get<1>(epoch).push_back((double)buf[i]);
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void train (int instanceIndex, int iterations, int startIndex) {

        Network* net = Network::getInstance(instanceIndex);

        if (iterations == -1) {
            net->train(net->trainingData.size(), 0);
        } else {
            net->train(iterations, startIndex);
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void loadTestingData (int instanceIndex, float *buf, int total, int size, int dimension) {
        Network* net = Network::getInstance(instanceIndex);
        net->testData.clear();
        std::tuple<std::vector<double>, std::vector<double> > epoch;

        // Push test data to memory
        for (int i=0; i<total; i++) {
            if (i && i%size==0) {
                net->testData.push_back(epoch);
                std::get<0>(epoch).clear();
                std::get<1>(epoch).clear();
            }

            if (i%size<dimension) {
                std::get<0>(epoch).push_back((double)buf[i]);
            } else {
                std::get<1>(epoch).push_back((double)buf[i]);
            }
        }

        net->testData.push_back(epoch);
    }

    EMSCRIPTEN_KEEPALIVE
    void shuffleTrainingData (int instanceIndex) {
        NetUtil::shuffle(Network::getInstance(instanceIndex)->trainingData);
    }

    EMSCRIPTEN_KEEPALIVE
    double test (int instanceIndex, int iterations, int startIndex) {

        Network* net = Network::getInstance(instanceIndex);

        double avgError;

        if (iterations == -1) {
            avgError = net->test(net->testData.size(), 0);
        } else {
            avgError = net->test(iterations, startIndex);
        }

        return avgError;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_miniBatchSize (int instanceIndex, int mbs) {
        Network::getInstance(instanceIndex)->miniBatchSize = mbs;
    }

    EMSCRIPTEN_KEEPALIVE
    void resetDeltaWeights (int instanceIndex) {
        Network::getInstance(instanceIndex)->resetDeltaWeights();
    }

    /* ConvLayer */
    EMSCRIPTEN_KEEPALIVE
    int get_conv_channels (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->channels;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_conv_channels (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->channels = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_conv_filterSize (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->filterSize;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_conv_filterSize (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->filterSize = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_conv_stride (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->stride;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_conv_stride (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->stride = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_conv_zeroPadding (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->zeroPadding;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_conv_zeroPadding (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->zeroPadding = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_conv_inMapValuesCount (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->inMapValuesCount;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_conv_inMapValuesCount (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->inMapValuesCount = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_conv_inZPMapValuesCount (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->inZPMapValuesCount;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_conv_inZPMapValuesCount (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->inZPMapValuesCount = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_conv_outMapSize (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->outMapSize;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_conv_outMapSize (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->outMapSize = value;
    }

    EMSCRIPTEN_KEEPALIVE
    void setConvActivation (int instanceIndex, int layerIndex, int activationFnIndex) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];
        layer->hasActivation = true;

        switch (activationFnIndex) {
            case 0:
                layer->activationF = &NetMath::sigmoid<Filter>;
                break;
            case 1:
                layer->activationF = &NetMath::tanh<Filter>;
                break;
            case 2:
                layer->activationF = &NetMath::lecuntanh<Filter>;
                break;
            case 3:
                layer->activationF = &NetMath::relu<Filter>;
                break;
            case 4:
                layer->activationF = &NetMath::lrelu<Filter>;
                break;
            case 5:
                layer->activationF = &NetMath::rrelu<Filter>;
                break;
            case 6:
                layer->activationF = &NetMath::elu<Filter>;
                break;
        }
    }

    /* Neuron */
    EMSCRIPTEN_KEEPALIVE
    double* get_weights (int instanceIndex, int layerIndex, int neuronIndex) {
        Network* net = Network::getInstance(instanceIndex);

        int neuronSize = net->layers[layerIndex]->neurons[neuronIndex]->weights.size();
        double weights[neuronSize];

        for (int i=0; i<neuronSize; i++) {
            weights[i] = net->layers[layerIndex]->neurons[neuronIndex]->weights[i];
        }

        auto ptr = &weights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_weights (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int w=0; w<bufSize; w++) {
            net->layers[layerIndex]->neurons[neuronIndex]->weights[w] = buf[w];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_bias (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->bias;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_bias (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->bias = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_deltaWeights (int instanceIndex, int layerIndex, int neuronIndex) {
        Network* net = Network::getInstance(instanceIndex);

        int neuronSize = net->layers[layerIndex]->neurons[neuronIndex]->deltaWeights.size();
        double deltaWeights[neuronSize];

        for (int i=0; i<neuronSize; i++) {
            deltaWeights[i] = net->layers[layerIndex]->neurons[neuronIndex]->deltaWeights[i];
        }

        auto ptr = &deltaWeights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_deltaWeights (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int dw=0; dw<bufSize; dw++) {
            net->layers[layerIndex]->neurons[neuronIndex]->deltaWeights[dw] = buf[dw];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_deltaBias (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->deltaBias;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_deltaBias (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->deltaBias = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_weightGain (int instanceIndex, int layerIndex, int neuronIndex) {
        Network* net = Network::getInstance(instanceIndex);

        int neuronSize = net->layers[layerIndex]->neurons[neuronIndex]->weightGain.size();
        double weightGain[neuronSize];

        for (int i=0; i<neuronSize; i++) {
            weightGain[i] = net->layers[layerIndex]->neurons[neuronIndex]->weightGain[i];
        }

        auto ptr = &weightGain[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_weightGain (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int dw=0; dw<bufSize; dw++) {
            net->layers[layerIndex]->neurons[neuronIndex]->weightGain[dw] = buf[dw];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_weightsCache (int instanceIndex, int layerIndex, int neuronIndex) {
        Network* net = Network::getInstance(instanceIndex);

        int neuronSize = net->layers[layerIndex]->neurons[neuronIndex]->weightsCache.size();
        double weightsCache[neuronSize];

        for (int i=0; i<neuronSize; i++) {
            weightsCache[i] = net->layers[layerIndex]->neurons[neuronIndex]->weightsCache[i];
        }

        auto ptr = &weightsCache[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_weightsCache (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int dw=0; dw<bufSize; dw++) {
            net->layers[layerIndex]->neurons[neuronIndex]->weightsCache[dw] = buf[dw];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void set_biasGain (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasGain = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_biasGain (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasGain;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_biasCache (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasCache = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_biasCache (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasCache;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_m (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->m = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_m (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->m;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_v (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->v = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_v (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->v;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_adadeltaBiasCache (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->adadeltaBiasCache = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_adadeltaBiasCache (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->adadeltaBiasCache;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_adadeltaCache (int instanceIndex, int layerIndex, int neuronIndex) {
        Network* net = Network::getInstance(instanceIndex);

        int neuronSize = net->layers[layerIndex]->neurons[neuronIndex]->adadeltaCache.size();
        double adadeltaCache[neuronSize];

        for (int i=0; i<neuronSize; i++) {
            adadeltaCache[i] = net->layers[layerIndex]->neurons[neuronIndex]->adadeltaCache[i];
        }

        auto ptr = &adadeltaCache[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_adadeltaCache (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int dw=0; dw<bufSize; dw++) {
            net->layers[layerIndex]->neurons[neuronIndex]->adadeltaCache[dw] = buf[dw];
        }
    }

    /* Filter */
    EMSCRIPTEN_KEEPALIVE
    double get_filter_bias (int instanceIndex, int layerIndex, int filterIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->bias;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_bias (int instanceIndex, int layerIndex, int filterIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->bias = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_weights (int instanceIndex, int layerIndex, int filterIndex) {

        Network* net = Network::getInstance(instanceIndex);
        Filter* filter = net->layers[layerIndex]->filters[filterIndex];

        int weightsDepth = filter->weights.size();
        int weightsSpan = filter->weights[0].size();
        double weights[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    weights[d*weightsSpan + r*weightsSpan + c] = filter->weights[d][r][c];
                }
            }
        }

        auto ptr = &weights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_weights (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Filter* filter = Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    filter->weights[d][r][c] = buf[d*rows*cols + r*cols + c];
                }
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_filter_deltaBias (int instanceIndex, int layerIndex, int filterIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->deltaBias;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_deltaBias (int instanceIndex, int layerIndex, int filterIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->deltaBias = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_deltaWeights (int instanceIndex, int layerIndex, int filterIndex) {

        Network* net = Network::getInstance(instanceIndex);
        Filter* filter = net->layers[layerIndex]->filters[filterIndex];

        int weightsDepth = filter->deltaWeights.size();
        int weightsSpan = filter->deltaWeights[0].size();
        double deltaWeights[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    deltaWeights[d*weightsSpan + r*weightsSpan + c] = filter->deltaWeights[d][r][c];
                }
            }
        }

        auto ptr = &deltaWeights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_deltaWeights (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Filter* filter = Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    filter->deltaWeights[d][r][c] = buf[d*rows*cols + r*cols + c];
                }
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_filter_biasGain (int instanceIndex, int layerIndex, int filterIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->biasGain;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_biasGain (int instanceIndex, int layerIndex, int filterIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->biasGain = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filterWeightGain (int instanceIndex, int layerIndex, int filterIndex) {

        Network* net = Network::getInstance(instanceIndex);
        Filter* filter = net->layers[layerIndex]->filters[filterIndex];

        int weightsDepth = filter->weights.size();
        int weightsSpan = filter->weights[0].size();
        double weights[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    weights[d*weightsSpan + r*weightsSpan + c] = filter->weightGain[d][r][c];
                }
            }
        }

        auto ptr = &weights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_weightGain (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Filter* filter = Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    filter->weightGain[d][r][c] = buf[d*rows*cols + r*cols + c];
                }
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_filter_biasCache (int instanceIndex, int layerIndex, int filterIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->biasCache;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_biasCache (int instanceIndex, int layerIndex, int filterIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->biasCache = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_weightsCache (int instanceIndex, int layerIndex, int filterIndex) {

        Network* net = Network::getInstance(instanceIndex);
        Filter* filter = net->layers[layerIndex]->filters[filterIndex];

        int weightsDepth = filter->weights.size();
        int weightsSpan = filter->weights[0].size();
        double weights[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    weights[d*weightsSpan + r*weightsSpan + c] = filter->weightsCache[d][r][c];
                }
            }
        }

        auto ptr = &weights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_weightsCache (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Filter* filter = Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    filter->weightsCache[d][r][c] = buf[d*rows*cols + r*cols + c];
                }
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_filter_adadeltaBiasCache (int instanceIndex, int layerIndex, int filterIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->adadeltaBiasCache;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_adadeltaBiasCache (int instanceIndex, int layerIndex, int filterIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->adadeltaBiasCache = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_adadeltaWeightsCache (int instanceIndex, int layerIndex, int filterIndex) {

        Network* net = Network::getInstance(instanceIndex);
        Filter* filter = net->layers[layerIndex]->filters[filterIndex];

        int weightsDepth = filter->weights.size();
        int weightsSpan = filter->weights[0].size();
        double weights[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    weights[d*weightsSpan + r*weightsSpan + c] = filter->adadeltaCache[d][r][c];
                }
            }
        }

        auto ptr = &weights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_adadeltaWeightsCache (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Filter* filter = Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    filter->adadeltaCache[d][r][c] = buf[d*rows*cols + r*cols + c];
                }
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_filter_m (int instanceIndex, int layerIndex, int filterIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->m;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_m (int instanceIndex, int layerIndex, int filterIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->m = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_filter_v (int instanceIndex, int layerIndex, int filterIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->v;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_v (int instanceIndex, int layerIndex, int filterIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex]->v = value;
    }
}