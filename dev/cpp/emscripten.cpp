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
    void setActivation (int instanceIndex, int activationFnIndex) {
        Network* net = Network::getInstance(instanceIndex);

        switch (activationFnIndex) {
            case 0:
                net->activation = &NetMath::sigmoid;
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
        }
    }

    EMSCRIPTEN_KEEPALIVE
    void set_rho  (int instanceIndex, float rho) {
        Network::getInstance(instanceIndex)->rho = rho;
    }

    float get_rho (int instanceIndex) {
        return Network::getInstance(instanceIndex)->rho;
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
        Network::getInstance(instanceIndex)->layers.push_back(new Layer(instanceIndex, size));
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
    void train (int instanceIndex, float *buf, int total, int size, int dimension) {

        Network* net = Network::getInstance(instanceIndex);
        net->trainingData.clear();

        std::tuple<std::vector<double>, std::vector<double> > epoch;

        // Push training data to memory
        for (int i=0; i<total; i++) {

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

        net->trainingData.push_back(epoch);
        net->train();

        net->trainingData.clear();
    }

    EMSCRIPTEN_KEEPALIVE
    double test (int instanceIndex, float *buf, int total, int size, int dimension) {

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

        double avgError = net->test();
        net->testData.clear();

        return avgError;
    }

    EMSCRIPTEN_KEEPALIVE
    void resetDeltaWeights (int instanceIndex) {
        Network::getInstance(instanceIndex)->resetDeltaWeights();
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
    double get_bias (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->bias;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_bias (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->bias = value;
    }

    EMSCRIPTEN_KEEPALIVE
    void set_biasGain (int instanceIndex, int layerIndex, int neuronIndex, double value) {
        Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasGain = value;
    }

    EMSCRIPTEN_KEEPALIVE
    double get_biasGain (int instanceIndex, int layerIndex, int neuronIndex) {
        return Network::getInstance(instanceIndex)->layers[layerIndex]->neurons[neuronIndex]->biasGain;
    }
}