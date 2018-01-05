
FCLayer::FCLayer (int netI, int s) : Layer(netI, s) {
    netInstance = netI;
    size = s;
    type = "FC";
    hasActivation = false;
}

FCLayer::~FCLayer (void) {
    for (int n=0; n<neurons.size(); n++) {
        delete neurons[n];
    }
}

void FCLayer::assignNext (Layer* l) {
    nextLayer = l;
}

void FCLayer::assignPrev (Layer* l) {
    prevLayer = l;
}

void FCLayer::init (int layerIndex) {
    for (int n=0; n<size; n++) {

        Neuron* neuron = new Neuron();

        int weightsCount = 0;

        if (layerIndex) {

            if (prevLayer->type == "FC") {
                weightsCount = prevLayer->size;
            } else if (prevLayer->type == "Conv") {
                weightsCount = prevLayer->filters.size() * prevLayer->outMapSize * prevLayer->outMapSize;
            } else {
                weightsCount = prevLayer->activations.size() * prevLayer->outMapSize * prevLayer->outMapSize;
            }

            weights.push_back(Network::getInstance(netInstance)->weightInitFn(netInstance, layerIndex, weightsCount));
            deltaWeights.push_back(std::vector<double>(weightsCount, 0));
            biases.push_back(1);
        }

        neuron->init(netInstance, weightsCount);
        neurons.push_back(neuron);

        sums.push_back(0);
        errs.push_back(0);
        actvns.push_back(0);
    }
}

void FCLayer::forward (void) {

    Network* net = Network::getInstance(netInstance);

    for (int n=0; n<neurons.size(); n++) {

        neurons[n]->dropped = (double) rand() / (RAND_MAX) > net->dropout;

        if (net->isTraining && neurons[n]->dropped) {
            actvns[n] = 0;

        } else {
            sums[n] = biases[n];

            if (prevLayer->type == "FC") {
                for (int pn=0; pn<prevLayer->neurons.size(); pn++) {
                    sums[n] += prevLayer->actvns[pn] * weights[n][pn];
                }
            } else if (prevLayer->type == "Conv") {

                std::vector<double> activations = NetUtil::getActivations(prevLayer);

                for (int ai=0; ai<activations.size(); ai++) {
                    sums[n] += activations[ai] * weights[n][ai];
                }
            } else {
                for (int c=0; c<prevLayer->channels; c++) {
                    for (int r=0; r<prevLayer->outMapSize; r++) {
                        for (int v=0; v<prevLayer->outMapSize; v++) {
                            sums[n] += prevLayer->activations[c][r][v] * weights[n][c * prevLayer->outMapSize * prevLayer->outMapSize + r * prevLayer->outMapSize + v ];
                        }
                    }
                }
            }

            if (hasActivation) {
                actvns[n] = activation(sums[n], false, neurons[n]) / net->dropout;
            } else {
                actvns[n] = sums[n] / net->dropout;
            }
        }
    }
}

void FCLayer::backward (bool lastLayer) {

    Network* net = Network::getInstance(netInstance);

    for (int n=0; n<neurons.size(); n++) {

        if (neurons[n]->dropped) {
            errs[n] = 0;
            neurons[n]->deltaBias = 0;

        } else {

            if (!lastLayer) {
                if (hasActivation) {
                    neurons[n]->derivative = activation(sums[n], true, neurons[n]);
                } else {
                    neurons[n]->derivative = 1;
                }

                double weightedErrors = 0.0;

                for (int nn=0; nn<nextLayer->neurons.size(); nn++) {
                    weightedErrors += nextLayer->errs[nn] * nextLayer->weights[nn][n];
                }

                errs[n] = neurons[n]->derivative * weightedErrors;
            }

            if (prevLayer->type == "FC") {
                for (int wi=0; wi<weights[n].size(); wi++) {
                    deltaWeights[n][wi] += errs[n] * prevLayer->actvns[wi];
                }

            } else {

                std::vector<double> activations = NetUtil::getActivations(prevLayer);

                for (int wi=0; wi<weights[n].size(); wi++) {
                    deltaWeights[n][wi] += errs[n] * activations[wi];
                }
            }

            neurons[n]->deltaBias += errs[n];
        }
    }
}

void FCLayer::resetDeltaWeights (void) {
    for(int n=0; n<neurons.size(); n++) {
        neurons[n]->deltaBias = 0;
        deltaWeights[n] = std::vector<double>(weights[n].size(), 0);
    }
}


void FCLayer::applyDeltaWeights (void) {

    Network* net = Network::getInstance(netInstance);

    for(int n=0; n<neurons.size(); n++) {
        for (int dw=0; dw<deltaWeights[n].size(); dw++) {
            if (net->l2) net->l2Error += 0.5 * net->l2 * pow(weights[n][dw], 2);
            if (net->l1) net->l1Error += net->l1 * fabs(weights[n][dw]);
        }
    }

    // Function pointers are far too slow for this
    // Using code repetitive switch statements makes a substantial perf difference
    // Doesn't mean I like it :(
    switch (net->updateFnIndex) {
        case 0: // vanilla
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<deltaWeights[n].size(); dw++) {

                    double regularized = (deltaWeights[n][dw]
                        + net->l2 * weights[n][dw]
                        + net->l1 * (weights[n][dw] > 0 ? 1 : -1)) / net->miniBatchSize;

                    weights[n][dw] = NetMath::vanillaupdatefn(netInstance, weights[n][dw], regularized);

                    if (net->maxNorm) net->maxNormTotal += weights[n][dw] * weights[n][dw];
                }
                biases[n] = NetMath::vanillaupdatefn(netInstance, biases[n], neurons[n]->deltaBias);
            }
            break;
        case 1: // gain
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<deltaWeights[n].size(); dw++) {

                    double regularized = (deltaWeights[n][dw]
                        + net->l2 * weights[n][dw]
                        + net->l1 * (weights[n][dw] > 0 ? 1 : -1)) / net->miniBatchSize;

                    weights[n][dw] = NetMath::gain(netInstance, weights[n][dw], deltaWeights[n][dw], neurons[n], dw);

                    if (net->maxNorm) net->maxNormTotal += weights[n][dw] * weights[n][dw];
                }
                biases[n] = NetMath::gain(netInstance, biases[n], neurons[n]->deltaBias, neurons[n], -1);
            }
            break;
        case 2: // adagrad
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<deltaWeights[n].size(); dw++) {

                    double regularized = (deltaWeights[n][dw]
                        + net->l2 * weights[n][dw]
                        + net->l1 * (weights[n][dw] > 0 ? 1 : -1)) / net->miniBatchSize;

                    weights[n][dw] = NetMath::adagrad(netInstance, weights[n][dw], regularized, neurons[n], dw);

                    if (net->maxNorm) net->maxNormTotal += weights[n][dw] * weights[n][dw];
                }
                biases[n] = NetMath::adagrad(netInstance, biases[n], neurons[n]->deltaBias, neurons[n], -1);
            }
            break;
        case 3: // rmsprop
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<deltaWeights[n].size(); dw++) {

                    double regularized = (deltaWeights[n][dw]
                        + net->l2 * weights[n][dw]
                        + net->l1 * (weights[n][dw] > 0 ? 1 : -1)) / net->miniBatchSize;

                    weights[n][dw] = NetMath::rmsprop(netInstance, weights[n][dw], regularized, neurons[n], dw);

                    if (net->maxNorm) net->maxNormTotal += weights[n][dw] * weights[n][dw];
                }
                biases[n] = NetMath::rmsprop(netInstance, biases[n], neurons[n]->deltaBias, neurons[n], -1);
            }
            break;
        case 4: // adam
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<deltaWeights[n].size(); dw++) {

                    double regularized = (deltaWeights[n][dw]
                        + net->l2 * weights[n][dw]
                        + net->l1 * (weights[n][dw] > 0 ? 1 : -1)) / net->miniBatchSize;

                    weights[n][dw] = NetMath::adam(netInstance, weights[n][dw], regularized, neurons[n], dw);

                    if (net->maxNorm) net->maxNormTotal += weights[n][dw] * weights[n][dw];
                }
                biases[n] = NetMath::adam(netInstance, biases[n], neurons[n]->deltaBias, neurons[n], -1);
            }
            break;
        case 5: // adadelta
            for(int n=0; n<neurons.size(); n++) {
                for (int dw=0; dw<deltaWeights[n].size(); dw++) {

                    double regularized = (deltaWeights[n][dw]
                        + net->l2 * weights[n][dw]
                        + net->l1 * (weights[n][dw] > 0 ? 1 : -1)) / net->miniBatchSize;

                    weights[n][dw] = NetMath::adadelta(netInstance, weights[n][dw], regularized, neurons[n], dw);

                    if (net->maxNorm) net->maxNormTotal += weights[n][dw] * weights[n][dw];
                }
                biases[n] = NetMath::adadelta(netInstance, biases[n], neurons[n]->deltaBias, neurons[n], -1);
            }
            break;
    }

    if (net->maxNorm) {
        net->maxNormTotal = sqrt(net->maxNormTotal);
        NetMath::maxNorm(netInstance);
    }
}
