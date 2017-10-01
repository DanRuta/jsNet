
void Neuron::init (int netInstance) {

    for (int i=0; i<weights.size(); i++) {
        deltaWeights.push_back(0);
    }

    deltaBias = 0;

    switch (Network::getInstance(netInstance)->updateFnIndex) {
        case 1: // gain
            biasGain = 1;
            for (int i=0; i<weights.size(); i++) {
                weightGain.push_back(1);
            }
            break;
        case 2: // adagrad
            biasCache = 0;
            for (int i=0; i<weights.size(); i++) {
                weightsCache.push_back(0);
            }
            break;
    }
}