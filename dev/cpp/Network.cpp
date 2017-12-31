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
            layers[l-1]->fanOut = layers[l]->size;
            layers[l]->assignPrev(layers[l-1]);
            layers[l]->fanIn = layers[l-1]->size;
        }

        layers[l]->init(l);
    }
}

std::vector<double> Network::forward (std::vector<double> input) {

    for (int v=0; v<input.size(); v++) {
        layers[0]->neurons[v]->activation = input[v];
    }

    for (int l=1; l<layers.size(); l++) {
        layers[l]->forward();
    }

    std::vector<double> output;

    for (int i=0; i<layers[layers.size()-1]->neurons.size(); i++) {
        output.push_back(layers[layers.size()-1]->neurons[i]->activation);
    }

    return output;
}

void Network::backward (std::vector<double> expected) {

    layers[layers.size()-1]->backward(expected);

    for (int l=layers.size()-2; l>0; l--) {
        std::vector<double> emptyVec;
        layers[l]->backward(emptyVec);
    }
}

void Network::train (int its, int startI) {

    double totalErrors = 0.0;
    double iterationError = 0.0;

    isTraining = true;

    for (int i=startI; i<(startI+its); i++) {

        iterations++;

        std::vector<double> output = forward(std::get<0>(trainingData[i]));
        iterationError = costFunction(std::get<1>(trainingData[i]), output);
        totalErrors += iterationError;

        backward(std::get<1>(trainingData[i]));

        if ((i+1) % miniBatchSize == 0) {
            applyDeltaWeights();
            resetDeltaWeights();
        } else if (i >= trainingData.size()) {
            applyDeltaWeights();
        }

    }

    isTraining = false;
    error = totalErrors / its;
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

Layer* Network::getLayer(int i) {
    return layers[i];
}

std::vector<Network*> Network::netInstances = {};
