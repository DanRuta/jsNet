#include <stdio.h>
#include <emscripten.h>
#include "Network.cpp"

int main(int argc, char const *argv[]) {

    EM_ASM(
        if (typeof window!='undefined') {
            window.dispatchEvent(new CustomEvent('jsNetWASMLoaded'));
            // https://github.com/DanRuta/jsNet/issues/33
            window.global = window.global || {};
        }

        global.onWASMLoaded && global.onWASMLoaded();
    );

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
    double getValidationError (int instanceIndex) {
        return Network::getInstance(instanceIndex)->validationError;
    }

    EMSCRIPTEN_KEEPALIVE
    double getLastValidationError (int instanceIndex) {
        return Network::getInstance(instanceIndex)->lastValidationError;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_iterations (int instanceIndex) {
        return Network::getInstance(instanceIndex)->iterations;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_iterations (int instanceIndex, float it) {
        Network::getInstance(instanceIndex)->iterations = it;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_validations (int instanceIndex) {
        return Network::getInstance(instanceIndex)->validations;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_validations (int instanceIndex, float v) {
        Network::getInstance(instanceIndex)->validations = v;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_validationInterval (int instanceIndex) {
        return Network::getInstance(instanceIndex)->validationInterval;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_validationInterval (int instanceIndex, float vr) {
        Network::getInstance(instanceIndex)->validationInterval = vr;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_stoppedEarly (int instanceIndex) {
        return Network::getInstance(instanceIndex)->stoppedEarly;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_stoppedEarly (int instanceIndex, int se) {
        Network::getInstance(instanceIndex)->stoppedEarly = se;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_trainingLogging (int instanceIndex) {
        return Network::getInstance(instanceIndex)->trainingLogging;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_trainingLogging (int instanceIndex, int tl) {
        Network::getInstance(instanceIndex)->trainingLogging = tl;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_earlyStoppingType (int instanceIndex) {
        return Network::getInstance(instanceIndex)->earlyStoppingType;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_earlyStoppingType (int instanceIndex, float est) {
        Network::getInstance(instanceIndex)->earlyStoppingType = est;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_earlyStoppingThreshold (int instanceIndex) {
        return Network::getInstance(instanceIndex)->earlyStoppingThreshold;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_earlyStoppingThreshold (int instanceIndex, float est) {
        Network::getInstance(instanceIndex)->earlyStoppingThreshold = est;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_earlyStoppingBestError (int instanceIndex) {
        return Network::getInstance(instanceIndex)->earlyStoppingBestError;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_earlyStoppingBestError (int instanceIndex, float esbe) {
        Network::getInstance(instanceIndex)->earlyStoppingBestError = esbe;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_earlyStoppingPatienceCounter (int instanceIndex) {
        return Network::getInstance(instanceIndex)->earlyStoppingPatienceCounter;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_earlyStoppingPatienceCounter (int instanceIndex, float espc) {
        Network::getInstance(instanceIndex)->earlyStoppingPatienceCounter = espc;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_earlyStoppingPatience (int instanceIndex) {
        return Network::getInstance(instanceIndex)->earlyStoppingPatience;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_earlyStoppingPatience (int instanceIndex, float esp) {
        Network::getInstance(instanceIndex)->earlyStoppingPatience = esp;
    }

    EMSCRIPTEN_KEEPALIVE
    float get_earlyStoppingPercent (int instanceIndex) {
        return Network::getInstance(instanceIndex)->earlyStoppingPercent;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_earlyStoppingPercent (int instanceIndex, float esp) {
        Network::getInstance(instanceIndex)->earlyStoppingPercent = esp;
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
    void addPoolLayer (int instanceIndex, int size) {
        Network::getInstance(instanceIndex)->layers.push_back(new PoolLayer(instanceIndex, size));
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

        double returnArr[activations.size()];
        for (int v=0; v<activations.size(); v++) {
            returnArr[v] = activations[v];
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
    void loadValidationData (int instanceIndex, float *buf, int total, int size, int dimension) {
        Network* net = Network::getInstance(instanceIndex);
        net->validationData.clear();

        std::tuple<std::vector<double>, std::vector<double> > epoch;

        // Push validation data to memory
        for (int i=0; i<=total; i++) {

            if (i && i%size==0) {
                net->validationData.push_back(epoch);
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
    void restoreValidation (int instanceIndex) {
        Network::getInstance(instanceIndex)->restoreValidation();
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

    /* FCLayer */
    EMSCRIPTEN_KEEPALIVE
    void set_fc_activation (int instanceIndex, int layerIndex, int activationFnIndex) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];
        layer->hasActivation = true;

        switch (activationFnIndex) {
            case -1:
                layer->hasActivation = false;
                break;
            case 0:
                layer->activation = &NetMath::sigmoid<Neuron>;
                break;
            case 1:
                layer->activation = &NetMath::tanh<Neuron>;
                break;
            case 2:
                layer->activation = &NetMath::lecuntanh<Neuron>;
                break;
            case 3:
                layer->activation = &NetMath::relu<Neuron>;
                break;
            case 4:
                layer->activation = &NetMath::lrelu<Neuron>;
                break;
            case 5:
                layer->activation = &NetMath::rrelu<Neuron>;
                break;
            case 6:
                layer->activation = &NetMath::elu<Neuron>;
                break;
        }
    }

    EMSCRIPTEN_KEEPALIVE
    int get_fc_activation (int instanceIndex, int layerIndex, int activationFnIndex) {
        // Do nothing
        return 0;
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
    void set_conv_activation (int instanceIndex, int layerIndex, int activationFnIndex) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];
        layer->hasActivation = true;

        switch (activationFnIndex) {
            case -1:
                layer->hasActivation = false;
                break;
            case 0:
                layer->activationC = &NetMath::sigmoid<Filter>;
                break;
            case 1:
                layer->activationC = &NetMath::tanh<Filter>;
                break;
            case 2:
                layer->activationC = &NetMath::lecuntanh<Filter>;
                break;
            case 3:
                layer->activationC = &NetMath::relu<Filter>;
                break;
            case 4:
                layer->activationC = &NetMath::lrelu<Filter>;
                break;
            case 5:
                layer->activationC = &NetMath::rrelu<Filter>;
                break;
            case 6:
                layer->activationC = &NetMath::elu<Filter>;
                break;
        }
    }

    EMSCRIPTEN_KEEPALIVE
    int get_conv_activation (int instanceIndex, int layerIndex, int activationFnIndex) {
        // Do nothing
        return 0;
    }


    /* PoolLayer */
    EMSCRIPTEN_KEEPALIVE
    void set_pool_activation (int instanceIndex, int layerIndex, int activationFnIndex) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];
        layer->hasActivation = true;

        switch (activationFnIndex) {
            case -1:
                layer->hasActivation = false;
                break;
            case 0:
                layer->activationP = &NetMath::sigmoid<Network>;
                break;
            case 1:
                layer->activationP = &NetMath::tanh<Network>;
                break;
            case 2:
                layer->activationP = &NetMath::lecuntanh<Network>;
                break;
            case 3:
                layer->activationP = &NetMath::relu<Network>;
                break;
            case 4:
                layer->activationP = &NetMath::lrelu<Network>;
                break;
            case 5:
                layer->activationP = &NetMath::rrelu<Network>;
                break;
            case 6:
                layer->activationP = &NetMath::elu<Network>;
                break;
        }
    }

    EMSCRIPTEN_KEEPALIVE
    int get_pool_activation (int instanceIndex, int layerIndex, int activationFnIndex) {
        // Do nothing
        return 0;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_pool_channels (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->channels;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_pool_channels (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->channels = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_pool_stride (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->stride;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_pool_stride (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->stride = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_pool_inMapValuesCount (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->inMapValuesCount;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_pool_inMapValuesCount (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->inMapValuesCount = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_pool_outMapSize (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->outMapSize;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_pool_outMapSize (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->outMapSize = value;
    }

    EMSCRIPTEN_KEEPALIVE
    int get_pool_prevLayerOutWidth (int instanceIndex, int layerIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->prevLayerOutWidth;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_pool_prevLayerOutWidth (int instanceIndex, int layerIndex, int value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->prevLayerOutWidth = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_pool_errors (int instanceIndex, int layerIndex) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        int mapDepth = layer->errors.size();
        int mapSpan = layer->errors[0].size();
        double errors[mapDepth * mapSpan * mapSpan];

        for (int d=0; d<mapDepth; d++) {
            for (int r=0; r<mapSpan; r++) {
                for (int c=0; c<mapSpan; c++) {
                    errors[d*mapSpan*mapSpan + r*mapSpan + c] = layer->errors[d][r][c];
                }
            }
        }

        auto ptr = &errors[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_pool_errors (int instanceIndex, int layerIndex, double *buf, int total, int depth, int rows, int cols) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    layer->errors[d][r][c] = buf[d*rows*cols + r*cols + c];
                }
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_pool_activations (int instanceIndex, int layerIndex) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        int mapDepth = layer->activations.size();
        int mapSpan = layer->activations[0].size();
        double activations[mapDepth * mapSpan * mapSpan];

        for (int d=0; d<mapDepth; d++) {
            for (int r=0; r<mapSpan; r++) {
                for (int c=0; c<mapSpan; c++) {
                    activations[d*mapSpan*mapSpan + r*mapSpan + c] = layer->activations[d][r][c];
                }
            }
        }

        auto ptr = &activations[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_pool_activations (int instanceIndex, int layerIndex, double *buf, int total, int depth, int rows, int cols) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    layer->activations[d][r][c] = buf[d*rows*cols + r*cols + c];
                }
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_pool_indeces (int instanceIndex, int layerIndex) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        int mapDepth = layer->indeces.size();
        int mapSpan = layer->indeces[0].size();
        double indeces[mapDepth * mapSpan * mapSpan];

        for (int d=0; d<mapDepth; d++) {
            for (int r=0; r<mapSpan; r++) {
                for (int c=0; c<mapSpan; c++) {
                    indeces[d*mapSpan*mapSpan + r*mapSpan + c] = layer->indeces[d][r][c][0]*2 + layer->indeces[d][r][c][1];
                }
            }
        }

        auto ptr = &indeces[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_pool_indeces (int instanceIndex, int layerIndex, double *buf, int total, int depth, int rows, int cols) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    layer->indeces[d][r][c][0] = (int) buf[r*cols + c] / 2;
                    layer->indeces[d][r][c][1] = (int) fmod(buf[d*rows*cols + r*cols + c], 2);
                }
            }
        }
    }


    /* Neuron */
    EMSCRIPTEN_KEEPALIVE
    double* get_neuron_weights (int instanceIndex, int layerIndex, int neuronIndex) {
        Network* net = Network::getInstance(instanceIndex);

        int neuronSize = net->layers[layerIndex]->weights[neuronIndex].size();
        double weights[neuronSize];

        for (int i=0; i<neuronSize; i++) {
            weights[i] = net->layers[layerIndex]->weights[neuronIndex][i];
        }

        auto ptr = &weights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_weights (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int w=0; w<bufSize; w++) {
            net->layers[layerIndex]->weights[neuronIndex][w] = buf[w];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_bias (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->biases[neuronIndex];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_bias (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->biases[neuronIndex] = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_neuron_deltaWeights (int instanceIndex, int layerIndex, int neuronIndex) {
        Network* net = Network::getInstance(instanceIndex);

        int neuronSize = net->layers[layerIndex]->deltaWeights[neuronIndex].size();
        double deltaWeights[neuronSize];

        for (int i=0; i<neuronSize; i++) {
            deltaWeights[i] = net->layers[layerIndex]->deltaWeights[neuronIndex][i];
        }

        auto ptr = &deltaWeights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_deltaWeights (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int dw=0; dw<bufSize; dw++) {
            net->layers[layerIndex]->deltaWeights[neuronIndex][dw] = buf[dw];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_deltaBias (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->deltaBiases[neuronIndex];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_deltaBias (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->deltaBiases[neuronIndex] = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_neuron_weightGain (int instanceIndex, int layerIndex, int neuronIndex) {
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
    void set_neuron_weightGain (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int dw=0; dw<bufSize; dw++) {
            net->layers[layerIndex]->neurons[neuronIndex]->weightGain[dw] = buf[dw];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_neuron_weightsCache (int instanceIndex, int layerIndex, int neuronIndex) {
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
    void set_neuron_weightsCache (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int dw=0; dw<bufSize; dw++) {
            net->layers[layerIndex]->neurons[neuronIndex]->weightsCache[dw] = buf[dw];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_biasGain (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasGain = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_biasGain (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasGain;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_biasCache (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasCache = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_biasCache (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasCache;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_m (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->m = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_m (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->m;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_v (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->v = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_v (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->v;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_adadeltaBiasCache (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->adadeltaBiasCache = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_adadeltaBiasCache (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->adadeltaBiasCache;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_neuron_adadeltaCache (int instanceIndex, int layerIndex, int neuronIndex) {
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
    void set_neuron_adadeltaCache (int instanceIndex, int layerIndex, int neuronIndex, double *buf, int bufSize) {
        Network* net = Network::getInstance(instanceIndex);

        for (int dw=0; dw<bufSize; dw++) {
            net->layers[layerIndex]->neurons[neuronIndex]->adadeltaCache[dw] = buf[dw];
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_sum (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->sums[neuronIndex];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_sum (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->sums[neuronIndex] = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_dropped (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->dropped;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_dropped (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->dropped = value==1;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_activation (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->actvns[neuronIndex];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_activation (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->actvns[neuronIndex] = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_error (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->errs[neuronIndex];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_error (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->errs[neuronIndex] = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_neuron_derivative (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->derivative;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_neuron_derivative (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->derivative = value;
    }

    /* Filter */
    EMSCRIPTEN_KEEPALIVE
    double get_filter_bias (int instanceIndex, int layerIndex, int filterIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->biases[filterIndex];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_bias (int instanceIndex, int layerIndex, int filterIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->biases[filterIndex] = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_weights (int instanceIndex, int layerIndex, int filterIndex) {

        Network* net = Network::getInstance(instanceIndex);
        Layer* layer = net->layers[layerIndex];

        int weightsDepth = layer->filterWeights[filterIndex].size();
        int weightsSpan = layer->filterWeights[filterIndex][0].size();
        double weights[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    weights[d*weightsSpan + r*weightsSpan + c] = layer->filterWeights[filterIndex][d][r][c];
                }
            }
        }

        auto ptr = &weights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_weights (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    layer->filterWeights[filterIndex][d][r][c] = buf[d*rows*cols + r*cols + c];
                }
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double get_filter_deltaBias (int instanceIndex, int layerIndex, int filterIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->deltaBiases[filterIndex];
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_deltaBias (int instanceIndex, int layerIndex, int filterIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->deltaBiases[filterIndex] = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_deltaWeights (int instanceIndex, int layerIndex, int filterIndex) {

        Network* net = Network::getInstance(instanceIndex);
        Layer* layer = net->layers[layerIndex];;

        int weightsDepth = layer->filterDeltaWeights[filterIndex].size();
        int weightsSpan = layer->filterDeltaWeights[filterIndex][0].size();
        double deltaWeights[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    deltaWeights[d*weightsSpan + r*weightsSpan + c] = layer->filterDeltaWeights[filterIndex][d][r][c];
                }
            }
        }

        auto ptr = &deltaWeights[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_deltaWeights (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        for (int d=0; d<depth; d++) {
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    layer->filterDeltaWeights[filterIndex][d][r][c] = buf[d*rows*cols + r*cols + c];
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
    double* get_filter_weightGain (int instanceIndex, int layerIndex, int filterIndex) {

        Network* net = Network::getInstance(instanceIndex);
        Filter* filter = net->layers[layerIndex]->filters[filterIndex];

        int weightsDepth = filter->weightGain.size();
        int weightsSpan = filter->weightGain[0].size();
        double weightGain[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    weightGain[d*weightsSpan + r*weightsSpan + c] = filter->weightGain[d][r][c];
                }
            }
        }

        auto ptr = &weightGain[0];
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

        int weightsDepth = filter->weightsCache.size();
        int weightsSpan = filter->weightsCache[0].size();
        double weightsCache[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    weightsCache[d*weightsSpan + r*weightsSpan + c] = filter->weightsCache[d][r][c];
                }
            }
        }

        auto ptr = &weightsCache[0];
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

        int weightsDepth = filter->adadeltaCache.size();
        int weightsSpan = filter->adadeltaCache[0].size();
        double adadeltaCache[weightsDepth * weightsSpan * weightsSpan];

        for (int d=0; d<weightsDepth; d++) {
            for (int r=0; r<weightsSpan; r++) {
                for (int c=0; c<weightsSpan; c++) {
                    adadeltaCache[d*weightsSpan + r*weightsSpan + c] = filter->adadeltaCache[d][r][c];
                }
            }
        }

        auto ptr = &adadeltaCache[0];
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

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_activationMap (int instanceIndex, int layerIndex, int filterIndex) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        int activationMapDepth = layer->activations[filterIndex].size();
        int activationMapSpan = layer->activations[filterIndex][0].size();
        double activationMap[activationMapDepth * activationMapSpan * activationMapSpan];

        for (int r=0; r<activationMapSpan; r++) {
            for (int c=0; c<activationMapSpan; c++) {
                activationMap[r*activationMapSpan + c] = layer->activations[filterIndex][r][c];
            }
        }

        auto ptr = &activationMap[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_activationMap (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        for (int r=0; r<rows; r++) {
            for (int c=0; c<cols; c++) {
                layer->activations[filterIndex][r][c] = buf[r*cols + c];
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_errorMap (int instanceIndex, int layerIndex, int filterIndex) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        int errorMapDepth = layer->errors[filterIndex].size();
        int errorMapSpan = layer->errors[filterIndex][0].size();
        double errorMap[errorMapDepth * errorMapSpan * errorMapSpan];

        for (int r=0; r<errorMapSpan; r++) {
            for (int c=0; c<errorMapSpan; c++) {
                errorMap[r*errorMapSpan + c] = layer->errors[filterIndex][r][c];
            }
        }

        auto ptr = &errorMap[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_errorMap (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Layer* layer = Network::getInstance(instanceIndex)->layers[layerIndex];

        for (int r=0; r<rows; r++) {
            for (int c=0; c<cols; c++) {
                layer->errors[filterIndex][r][c] = buf[r*cols + c];
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_sumMap (int instanceIndex, int layerIndex, int filterIndex) {

        Filter* filter = Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex];

        int sumMapDepth = filter->sumMap.size();
        int sumMapSpan = filter->sumMap[0].size();
        double sumMap[sumMapDepth * sumMapSpan * sumMapSpan];

        for (int r=0; r<sumMapSpan; r++) {
            for (int c=0; c<sumMapSpan; c++) {
                sumMap[r*sumMapSpan + c] = filter->sumMap[r][c];
            }
        }

        auto ptr = &sumMap[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_sumMap (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Filter* filter = Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex];

        for (int r=0; r<rows; r++) {
            for (int c=0; c<cols; c++) {
                filter->sumMap[r][c] = buf[r*cols + c];
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    double* get_filter_dropoutMap (int instanceIndex, int layerIndex, int filterIndex) {

        Filter* filter = Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex];

        int dropoutMapDepth = filter->dropoutMap.size();
        int dropoutMapSpan = filter->dropoutMap[0].size();
        double dropoutMap[dropoutMapDepth * dropoutMapSpan * dropoutMapSpan];

        for (int r=0; r<dropoutMapSpan; r++) {
            for (int c=0; c<dropoutMapSpan; c++) {
                dropoutMap[r*dropoutMapSpan + c] = filter->dropoutMap[r][c];
            }
        }

        auto ptr = &dropoutMap[0];
        return ptr;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_filter_dropoutMap (int instanceIndex, int layerIndex, int filterIndex, double *buf, int total, int depth, int rows, int cols) {

        Filter* filter = Network::getInstance(instanceIndex)->layers[layerIndex]->filters[filterIndex];

        for (int r=0; r<rows; r++) {
            for (int c=0; c<cols; c++) {
                filter->dropoutMap[r][c] = buf[r*cols + c]==1;
            }
        }
    }
}