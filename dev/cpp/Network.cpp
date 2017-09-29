#include "jsNet.h"
#include "Layer.cpp"
#include "Neuron.cpp"
#include "NetMath.cpp"

Network::~Network () {
    for (int l=0; l<layers.size(); l++) {
        delete layers[l];
    }
}

int Network::newNetwork(void) {
    Network* net = new Network();
    netInstances.push_back(net);
    return netInstances.size()-1;
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
        layers[l]->activation = activation;

        // Join layer
        if (l>0) {
            layers[l-1]->assignNext(layers[l]);
            layers[l]->assignPrev(layers[l-1]);
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
        layers[l]->backward((std::vector<double>){});
    }
}

void Network::train (void) {

    printf("Training...\n");

    double totalErrors = 0.0;

    for (int i=0; i<trainingData.size(); i++) {
        printf("Doing iteration\n");
        resetDeltaWeights();

        std::vector<double> output = forward(std::get<0>(trainingData[i]));
        totalErrors += costFunction(std::get<1>(trainingData[i]), output);

        backward(std::get<1>(trainingData[i]));
        applyDeltaWeights();
    }
}

double Network::test (void) {

    printf("Testing...\n");

    double totalErrors = 0.0;

    for (int i=0; i<testData.size(); i++) {
        std::vector<double> output = forward(std::get<0>(testData[i]));
        totalErrors += costFunction(std::get<1>(testData[i]), output);
    }

    return totalErrors / testData.size();
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
