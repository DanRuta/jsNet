#include "jsNet.h"
#include "FCLayer.cpp"
#include "ConvLayer.cpp"
#include "PoolLayer.cpp"
#include "Neuron.cpp"
#include "Filter.cpp"
#include "NetMath.cpp"
#include "NetUtil.cpp"

Network::~Network () {
    for (int l=0; l<layers.size(); l++) {
        delete layers[l];
    }
}

int Network::newNetwork(void) {
    Network* net = new Network();
    net->iterations = 0;
    net->rreluSlope = ((double) rand() / (RAND_MAX)) * 0.001;
    netInstances.push_back(net);
    net->instanceIndex = netInstances.size()-1;
    return net->instanceIndex;
}

void Network::deleteNetwork(void)  {
    std::vector<Network*> clearNetworkInstances;
    netInstances.swap(clearNetworkInstances);
}

void Network::deleteNetwork(int index) {
    delete netInstances[index];
    netInstances[index] = 0;
}

Network* Network::getInstance(int i) {
    return netInstances[i];
}

void Network::joinLayers(void) {
    for (int l=0; l<layers.size(); l++) {

        layers[l]->fanIn = -1;
        layers[l]->fanOut = -1;

        // Join layer
        if (l>0) {
            layers[l-1]->assignNext(layers[l]);

            if (l<layers.size()-1) {
                layers[l]->fanOut = layers[l+1]->size;
            }

            layers[l]->assignPrev(layers[l-1]);
            layers[l]->fanIn = layers[l-1]->size;
        } else {
            layers[0]->fanOut = layers[1]->size;
        }

        layers[l]->init(l);
    }
}

std::vector<double> Network::forward (std::vector<double> input) {

    layers[0]->actvns = input;

    for (int l=1; l<layers.size(); l++) {
        layers[l]->forward();
    }

    std::vector<double> output = layers[layers.size()-1]->sums;
    return output.size() > 1 ? NetMath::softmax(output) : output;
}

void Network::backward () {

    layers[layers.size()-1]->backward(true);

    for (int l=layers.size()-2; l>0; l--) {
        layers[l]->backward(false);
    }
}

void Network::train (int its, int startI) {

    double totalErrors = 0.0;
    double iterationError = 0.0;

    isTraining = true;
    validationError = 0;

    for (int iterationIndex=startI; iterationIndex<(startI+its); iterationIndex++) {

        iterations++;
        std::vector<double> output = forward(std::get<0>(trainingData[iterationIndex]));

        for (int n=0; n<output.size(); n++) {
            layers[layers.size()-1]->errs[n] = (std::get<1>(trainingData[iterationIndex])[n]==1 ? 1 : 0) - output[n];
        }

        if (validationInterval!=0 && iterationIndex!=0 && iterationIndex%validationInterval==0) {
            validationError = validate();

            if (earlyStoppingType && checkEarlyStopping()) {
                if (trainingLogging) {
                    printf("Stopping early\n");
                }
                stoppedEarly = true;
                break;
            }
        }

        backward();

        iterationError = costFunction(std::get<1>(trainingData[iterationIndex]), output);
        totalErrors += iterationError;

        if ((iterationIndex+1) % miniBatchSize == 0) {
            applyDeltaWeights();
            resetDeltaWeights();
        } else if (iterationIndex >= trainingData.size()) {
            applyDeltaWeights();
        }
    }

    isTraining = false;
    error = totalErrors / its;
}

double Network::validate (void) {

    double totalValidationErrors = 0;

    for (int i=0; i<validationData.size(); i++) {
        std::vector<double> output = forward(std::get<0>(validationData[i]));
        totalValidationErrors += costFunction(std::get<1>(validationData[i]), output);

        validations++;
    }
    lastValidationError = totalValidationErrors / validationData.size();
    return lastValidationError;
}

bool Network::checkEarlyStopping (void) {

    bool stop = false;

    switch (earlyStoppingType) {
        // threshold
        case 1:
            stop = lastValidationError <= earlyStoppingThreshold;

            // Do the last backward pass
            if (stop) {
                backward();
                applyDeltaWeights();
            }

            return stop;
        // patience
        case 2:

            if (lastValidationError < earlyStoppingBestError) {
                earlyStoppingPatienceCounter = 0;
                earlyStoppingBestError = lastValidationError;

                for (int l=1; l<layers.size(); l++) {
                    layers[l]->backUpValidation();
                }
            } else {
                earlyStoppingPatienceCounter++;
                stop = earlyStoppingPatienceCounter >= earlyStoppingPatience;
            }

            return stop;

        // divergence
        case 3:

            if (lastValidationError < earlyStoppingBestError) {

                earlyStoppingBestError = lastValidationError;

                for (int l=1; l<layers.size(); l++) {
                    layers[l]->backUpValidation();
                }

            } else {
                stop = (lastValidationError / earlyStoppingBestError) >= (1+earlyStoppingPercent/100);
            }

            return stop;

    }
    return stop;
}

double Network::test (int its, int startI) {

    double totalErrors = 0.0;

    for (int i=startI; i<(startI+its); i++) {
        std::vector<double> output = forward(std::get<0>(testData[i]));
        totalErrors += costFunction(std::get<1>(testData[i]), output);
    }

    return totalErrors / its;
}

void Network::resetDeltaWeights (void) {
    for (int l=1; l<layers.size(); l++) {
        layers[l]->resetDeltaWeights();
    }
}

void Network::applyDeltaWeights (void) {
    for (int l=1; l<layers.size(); l++) {
        layers[l]->applyDeltaWeights();
    }
}

void Network::restoreValidation (void) {
    for (int l=1; l<layers.size(); l++) {
        layers[l]->restoreValidation();
    }
}

std::vector<Network*> Network::netInstances = {};
