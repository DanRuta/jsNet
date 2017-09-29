
void Neuron::init (void) {

    for (int i=0; i<weights.size(); i++) {
        deltaWeights.push_back(0);
    }

    deltaBias = 0;
}