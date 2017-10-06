// Activation functions
double NetMath::sigmoid(double value, bool prime, Neuron* neuron) {
    double val = 1 / (1+exp(-value));
    return prime ? val*(1-val)
                 : val;
}

double NetMath::tanh(double value, bool prime, Neuron* neuron) {
    double ex = exp(2*value);
    double val = prime ? 4 / pow(exp(value)+exp(-value), 2) : (ex-1)/(ex+1);
    return val==0 ? 1e-18 : val;
}

double NetMath::lecuntanh(double value, bool prime, Neuron* neuron) {
  return prime ? 1.15333 * pow(NetMath::sech((2.0/3.0) * value), 2)
               : 1.7159 * NetMath::tanh((2.0/3.0) * value, false, neuron);
}

double NetMath::relu(double value, bool prime, Neuron* neuron) {
    return prime ? (value > 0 ? 1 : 0)
                 : (value>=0 ? value : 0);
}

double NetMath::lrelu(double value, bool prime, Neuron* neuron) {
    return prime ? value > 0 ? 1 : neuron->lreluSlope
                 : fmax(neuron->lreluSlope * fabs(value), value);
}

double NetMath::rrelu(double value, bool prime, Neuron* neuron) {
    return prime ? value > 0 ? 1 : neuron->rreluSlope
                 : fmax(neuron->rreluSlope, value);
}

double NetMath::elu(double value, bool prime, Neuron* neuron) {
    return prime ? value >= 0 ? 1 : elu(value, false, neuron) + neuron->eluAlpha
                 : value >= 0 ? value : neuron->eluAlpha * (exp(value) - 1);
}

// Cost Functions
double NetMath::meansquarederror (std::vector<double> calculated, std::vector<double> desired) {
    double error = 0.0;

    for (int v=0; v<calculated.size(); v++) {
        error += pow(calculated[v] - desired[v], 2);
    }

    return error / calculated.size();
}

double NetMath::crossentropy (std::vector<double> target, std::vector<double> output) {
    double error = 0.0;

    for (int v=0; v<target.size(); v++) {
        error -= target[v] * log(output[v]+1e-15) + ((1 - target[v]) * log(1+1e-15-output[v]));
    }

    return error;
}

// Weight update functions
double NetMath::vanillaupdatefn (int netInstance, double value, double deltaValue) {
    Network* net = Network::getInstance(netInstance);
    return value + net->learningRate * deltaValue;
}

double NetMath::gain(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex) {

    Network* net = Network::getInstance(netInstance);
    double newVal = value + net->learningRate * deltaValue * (weightIndex < 0 ? neuron->biasGain : neuron->weightGain[weightIndex]);

    if ((newVal<=0 && value>0) || (newVal>=0 && value<0)) {
        if (weightIndex>-1) {
            neuron->weightGain[weightIndex] = fmax(neuron->weightGain[weightIndex]*0.95, 0.5);
        } else {
            neuron->biasGain = fmax(neuron->biasGain*0.95, 0.5);
        }
    } else {
        if (weightIndex>-1) {
            neuron->weightGain[weightIndex] = fmin(neuron->weightGain[weightIndex]+0.05, 5);
        } else {
            neuron->biasGain = fmin(neuron->biasGain+0.05, 5);
        }
    }

    return newVal;
}

double NetMath::adagrad(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex) {

    if (weightIndex>-1) {
        neuron->weightsCache[weightIndex] += pow(deltaValue, 2);
    } else {
        neuron->biasCache += pow(deltaValue, 2);
    }

    Network* net = Network::getInstance(netInstance);
    return value + net->learningRate * deltaValue / (1e-6 + sqrt(weightIndex>-1 ? neuron->weightsCache[weightIndex]
                                                                                : neuron->biasCache));
}

double NetMath::rmsprop(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex) {

    Network* net = Network::getInstance(netInstance);

    if (weightIndex>-1) {
        neuron->weightsCache[weightIndex] = net->rmsDecay * neuron->weightsCache[weightIndex] + (1 - net->rmsDecay) * pow(deltaValue, 2);
    } else {
        neuron->biasCache = net->rmsDecay * neuron->biasCache + (1 - net->rmsDecay) * pow(deltaValue, 2);
    }

    return value + net->learningRate * deltaValue / (1e-6 + sqrt(weightIndex>-1 ? neuron->weightsCache[weightIndex]
                                                                                : neuron->biasCache));
}

double NetMath::adam(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex) {

    Network* net = Network::getInstance(netInstance);

    neuron->m = 0.9 * neuron->m + (1-0.9) * deltaValue;
    double mt = neuron->m / (1 - pow(0.9, net->iterations + 1));

    neuron->v = 0.999 * neuron->v + (1-0.999) * pow(deltaValue, 2);
    double vt = neuron->v / (1 - pow(0.999, net->iterations + 1));

    return value + net->learningRate * mt / (sqrt(vt) + 1e-6);
}

double NetMath::adadelta(int netInstance, double value, double deltaValue, Neuron* neuron, int weightIndex) {

    double rho = Network::getInstance(netInstance)->rho;

    if (weightIndex>-1) {
        neuron->weightsCache[weightIndex] = rho * neuron->weightsCache[weightIndex] + (1-rho) * pow(deltaValue, 2);
        double newVal = value + sqrt((neuron->adadeltaCache[weightIndex] + 1e-6) / (neuron->weightsCache[weightIndex] + 1e-6)) * deltaValue;
        neuron->adadeltaCache[weightIndex] = rho * neuron->adadeltaCache[weightIndex] + (1-rho) * pow(deltaValue, 2);
        return newVal;
    } else {
        neuron->biasCache = rho * neuron->biasCache + (1-rho) * pow(deltaValue, 2);
        double newVal = value + sqrt((neuron->adadeltaBiasCache + 1e-6) / (neuron->biasCache + 1e-6)) * deltaValue;
        neuron->adadeltaBiasCache = rho * neuron->adadeltaBiasCache + (1-rho) * pow(deltaValue, 2);
        return newVal;
    }
}

// Weights init
std::vector<double> NetMath::uniform (int netInstance, int size) {
    std::vector<double> values;

    float limit = Network::getInstance(netInstance)->weightsConfig["limit"];

    for (int v=0; v<size; v++) {
        values.push_back( (double) rand() / (RAND_MAX) * 2 * limit - limit );
    }

    return values;
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

double NetMath::sech(double value) {
    return (2 * exp(-value)) / (1+exp(-2*value));
}

void NetMath::maxNorm(int netInstance) {
    Network* net = Network::getInstance(netInstance);

    if (net->maxNormTotal > net->maxNorm) {

        double multiplier = net->maxNorm / (1e-18 + net->maxNormTotal);

        for (int l=1; l<net->layers.size(); l++) {
            for (int n=0; n<net->layers[l]->neurons.size(); n++) {
                for (int w=0; w<net->layers[l]->neurons[n]->weights.size(); w++) {
                    net->layers[l]->neurons[n]->weights[w] *= multiplier;
                }
            }
        }
    }

    net->maxNormTotal = 0;
}