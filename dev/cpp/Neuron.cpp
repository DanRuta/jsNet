
void Neuron::init (int netInstance) {

    for (int i=0; i<weights.size(); i++) {
        deltaWeights.push_back(0);
    }

    deltaBias = 0;

    Network* net = Network::getInstance(netInstance);

    switch (net->updateFnIndex) {
        case 1: // gain
            biasGain = 1;
            for (int i=0; i<weights.size(); i++) {
                weightGain.push_back(1);
            }
            break;
        case 2: // adagrad
        case 3: // rmsprop
        case 5: // adadelta
            biasCache = 0;
            for (int i=0; i<weights.size(); i++) {
                weightsCache.push_back(0);
            }

            if (net->updateFnIndex == 5) {
                adadeltaBiasCache = 0;
                for (int i=0; i<weights.size(); i++) {
                    adadeltaCache.push_back(0);
                }
            }
            break;

        case 4: // adam
            m = 0;
            v = 0;
            break;
    }

    if (net->activation == &NetMath::lrelu<Neuron>) {
        lreluSlope = net->lreluSlope;
    } else if (net->activation == &NetMath::rrelu<Neuron>) {
        rreluSlope = ((double) rand() / (RAND_MAX)) * 0.001;
    } else if (net->activation == &NetMath::elu<Neuron>) {
        eluAlpha = net->eluAlpha;
    }
}