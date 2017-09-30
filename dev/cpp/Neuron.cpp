
void Neuron::init (int netInstance) {

    for (int i=0; i<weights.size(); i++) {
        deltaWeights.push_back(0);
    }

    deltaBias = 0;

    switch (Network::getInstance(netInstance)->updateFnIndex) {
        case 1:
            biasGain = 1;
            for (int i=0; i<weights.size(); i++) {
                weightGain.push_back(1);
            }
            break;
    }
}