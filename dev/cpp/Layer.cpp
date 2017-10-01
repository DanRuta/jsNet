
Layer::Layer(int netI, int s) {
    netInstance = netI;
    size = s;
}

Layer::~Layer () {
    for (int n=0; n<neurons.size(); n++) {
        delete neurons[n];
    }
}

void Layer::assignNext (Layer* l) {
    nextLayer = l;
}

void Layer::assignPrev (Layer* l) {
    prevLayer = l;
}

void Layer::init (int layerIndex) {
    for (int n=0; n<size; n++) {
        Neuron* neuron = new Neuron();

        if (layerIndex) {

            for (int pn=0; pn<prevLayer->size; pn++) {
                // neuron->weights.push_back(rand()/1e+11-);
                neuron->weights.push_back(((double) rand() / (RAND_MAX))/5 - 0.1);
            }

            neuron->bias = ((double) rand() / (RAND_MAX))/5 - 0.1;
        }

        neuron->init(netInstance);
        neurons.push_back(neuron);
    }
}

void Layer::forward (void) {
    for (int n=0; n<neurons.size(); n++) {
        neurons[n]->sum = neurons[n]->bias;

        for (int pn=0; pn<prevLayer->neurons.size(); pn++) {
            neurons[n]->sum += prevLayer->neurons[pn]->activation * neurons[n]->weights[pn];
        }

        neurons[n]->activation = activation(neurons[n]->sum, false, neurons[n]);
    }
}

void Layer::backward (std::vector<double> expected) {
    for (int n=0; n<neurons.size(); n++) {

        if (expected.size()) {
            neurons[n]->error = expected[n] - neurons[n]->activation;
        } else {
            neurons[n]->derivative = activation(neurons[n]->sum, true, neurons[n]);

            double weightedErrors = 0.0;

            for (int nn=0; nn<nextLayer->neurons.size(); nn++) {
                weightedErrors += nextLayer->neurons[nn]->error * nextLayer->neurons[nn]->weights[n];
            }

            neurons[n]->error = neurons[n]->derivative * weightedErrors;
        }

        for (int wi=0; wi<neurons[n]->weights.size(); wi++) {
            neurons[n]->deltaWeights[wi] += neurons[n]->error * prevLayer->neurons[wi]->activation; //* (1  neurons[n]->deltaWeights[dw]);
        }

        neurons[n]->deltaBias = neurons[n]->error;
    }
}

void Layer::applyDeltaWeights (void) {

    // Function pointers are far too slow, here.
    // Using code repetitive switch statements makes a substantial perf difference
    int updateFnIndex = Network::getInstance(netInstance)->updateFnIndex;

    switch (updateFnIndex) {
        case 0: // vanilla
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<neurons[n]->deltaWeights.size(); dw++) {
                    neurons[n]->weights[dw] = NetMath::vanillaupdatefn(netInstance, neurons[n]->weights[dw], neurons[n]->deltaWeights[dw]);
                }
                neurons[n]->bias = NetMath::vanillaupdatefn(netInstance, neurons[n]->bias, neurons[n]->deltaBias);
            }
            break;
        case 1: // gain
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<neurons[n]->deltaWeights.size(); dw++) {
                    neurons[n]->weights[dw] = NetMath::gain(netInstance, neurons[n]->weights[dw], neurons[n]->deltaWeights[dw], neurons[n], dw);
                }
                neurons[n]->bias = NetMath::gain(netInstance, neurons[n]->bias, neurons[n]->deltaBias, neurons[n], -1);
            }
            break;
        case 2: // adagrad
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<neurons[n]->deltaWeights.size(); dw++) {
                    neurons[n]->weights[dw] = NetMath::adagrad(netInstance, neurons[n]->weights[dw], neurons[n]->deltaWeights[dw], neurons[n], dw);
                }
                neurons[n]->bias = NetMath::adagrad(netInstance, neurons[n]->bias, neurons[n]->deltaBias, neurons[n], -1);
            }
            break;
        case 3: // rmsprop
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<neurons[n]->deltaWeights.size(); dw++) {
                    neurons[n]->weights[dw] = NetMath::rmsprop(netInstance, neurons[n]->weights[dw], neurons[n]->deltaWeights[dw], neurons[n], dw);
                }
                neurons[n]->bias = NetMath::rmsprop(netInstance, neurons[n]->bias, neurons[n]->deltaBias, neurons[n], -1);
            }
            break;
    }
}

void Layer::resetDeltaWeights (void) {
    for(int n=0; n<neurons.size(); n++) {
        for (int dw=0; dw<neurons[n]->deltaWeights.size(); dw++) {
            neurons[n]->deltaWeights[dw] = 0;
        }
    }
}
