// Activation functions
double NetMath::sigmoid(double value, bool prime, Neuron* neuron) {
    double val = 1 / (1+exp(-value));
    return prime ? val*(1-val)
                 : val;
}

// Cost Functions
double NetMath::meansquarederror (std::vector<double> calculated, std::vector<double> desired) {
    double error = 0.0;

    for (int v=0; v<calculated.size(); v++) {
        error += pow(calculated[v] - desired[v], 2);
    }

    return error / calculated.size();
}

// Weight update functions
float NetMath::vanillaupdatefn (int netInstance, float value, float deltaValue) {
    Network* net = Network::getInstance(netInstance);
    return value + net->learningRate * deltaValue;
}

// Other
std::vector<double> NetMath::softmax (std::vector<double> values) {
    double total = 0.0;

    for (int i=0; i<values.size(); i++) {
        total += values[i];
    }

    for (int i=0; i<values.size(); i++) {
        if (total) {
            values[i] /= total;
        }
    }

    return values;
}