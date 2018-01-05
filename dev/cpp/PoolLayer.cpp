
PoolLayer::PoolLayer (int netI, int s) : Layer(netI, s) {
    netInstance = netI;
    size = s;
    type = "Pool";
    hasActivation = false;
}

PoolLayer::~PoolLayer (void) {}

void PoolLayer::assignNext (Layer* l) {
    nextLayer = l;
}

void PoolLayer::assignPrev (Layer* l) {
    prevLayer = l;
}

void PoolLayer::init (int layerIndex) {
    prevLayerOutWidth = sqrt(inMapValuesCount);
    activations = NetUtil::createVolume(channels, outMapSize, outMapSize, 0.0);
    errors = NetUtil::createVolume(channels, prevLayerOutWidth, prevLayerOutWidth, 0.0);
    std::vector<int> emptyIndex = {0,0};
    indeces = NetUtil::createVolume(channels, outMapSize, outMapSize, emptyIndex);
}

void PoolLayer::forward (void) {

    for (int channel=0; channel<channels; channel++) {

        NetMath::maxPool(this, channel);

        // Apply activations
        if (hasActivation) {
            for (int r=0; r<outMapSize; r++) {
                for (int v=0; v<outMapSize; v++) {
                    activations[channel][r][v] = activationP(activations[channel][r][v], false, Network::getInstance(netInstance));
                }
            }
        }
    }
}

void PoolLayer::backward (void) {

    // Clear the existing error values, first
    for (int c=0; c<channels; c++) {
        for (int r=0; r<prevLayerOutWidth; r++) {
            for (int v=0; v<prevLayerOutWidth; v++) {
                errors[c][r][v] = 0;
            }
        }
    }

    if (nextLayer->type=="FC") {

        for (int c=0; c<channels; c++) {
            for (int r=0; r<outMapSize; r++) {
                for (int v=0; v<outMapSize; v++) {

                    int rowI = indeces[c][r][v][0] + r * stride;
                    int colI = indeces[c][r][v][1] + v * stride;
                    int weightI = c * outMapSize*outMapSize + r * outMapSize + v;

                    for (int n=0; n<nextLayer->neurons.size(); n++) {
                        errors[c][rowI][colI] += nextLayer->neurons[n]->error * nextLayer->weights[n][weightI];
                    }
                }
            }
        }

    } else if (nextLayer->type=="Conv") {

        for (int c=0; c<channels; c++) {

            // Convolve on the error map
            std::vector<std::vector<double> > errs = NetUtil::buildConvErrorMap(outMapSize+nextLayer->zeroPadding*2, nextLayer, c);

            for (int r=0; r<outMapSize; r++) {
                for (int v=0; v<outMapSize; v++) {
                    int rowI = indeces[c][r][v][0] + r * stride;
                    int colI = indeces[c][r][v][1] + v * stride;

                    errors[c][rowI][colI] += errs[r][v];
                }
            }
        }

    } else {

        for (int c=0; c<channels; c++) {
            for (int r=0; r<outMapSize; r++) {
                for (int v=0; v<outMapSize; v++) {

                    int rowI = indeces[c][r][v][0] + r * stride;
                    int colI = indeces[c][r][v][1] + v * stride;

                    errors[c][rowI][colI] += nextLayer->errors[c][r][v];
                }
            }
        }
    }

    // Apply derivatives
    if (hasActivation) {
        for (int c=0; c<channels; c++) {
            for (int r=0; r<outMapSize; r++) {
                for (int v=0; v<outMapSize; v++) {

                    int rowI = indeces[c][r][v][0] + r * stride;
                    int colI = indeces[c][r][v][1] + v * stride;

                    errors[c][rowI][colI] *= activationP(errors[c][rowI][colI], true, Network::getInstance(netInstance));
                }
            }
        }
    }
}
