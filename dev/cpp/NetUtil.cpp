
void NetUtil::shuffle (std::vector<std::tuple<std::vector<double>, std::vector<double> > > &values) {
    for (int i=values.size(); i; i--) {
        int j = floor(rand() / RAND_MAX * i);
        std::tuple<std::vector<double>, std::vector<double> > x = values[i-1];
        values[i-1] = values[j];
        values[j] = x;
    }
}

std::vector<std::vector<double> > NetUtil::addZeroPadding (std::vector<std::vector<double> > map, int zP) {

    // Left and right columns
    for (int row=0; row<map.size(); row++) {
        for (int z=0; z<zP; z++) {
            map[row].insert(map[row].begin(), 0.0);
            map[row].push_back(0);
        }
    }

    // Top rows
    for (int z=0; z<zP; z++) {
        std::vector<double> row;

        for (int i=0; i<map[0].size(); i++) {
            row.push_back(0);
        }

        map.insert(map.begin(), row);
    }

    // Bottom rows
    for (int z=0; z<zP; z++) {
        std::vector<double> row;

        for (int i=0; i<map[0].size(); i++) {
            row.push_back(0);
        }

        map.push_back(row);
    }

    return map;
}

std::vector<std::vector<double> > NetUtil::arrayToMap (std::vector<double> array, int size) {

    std::vector<std::vector<double> > map;

    for (int i=0; i<size; i++) {
        std::vector<double> row;

        for (int j=0; j<size; j++) {
            row.push_back(array[i*size+j]);
        }

        map.push_back(row);
    }

    return map;
}

std::vector<std::vector<std::vector<double> > > NetUtil::arrayToVolume (std::vector<double> array, int channels) {

    std::vector<std::vector<std::vector<double> > > vol;
    int size = sqrt(array.size() / channels);
    int mapValues = size * size;

    for (int d=0; d<floor(array.size() / mapValues); d++) {
        std::vector<std::vector<double> > map;

        for (int i=0; i<size; i++) {
            std::vector<double> row;

            for (int j=0; j<size; j++) {
                row.push_back(array[d*mapValues + i*size+j]);
            }
            map.push_back(row);
        }

        vol.push_back(map);
    }

    return vol;
}

std::vector<std::vector<double> > NetUtil::convolve(std::vector<double> input, int zP,
    std::vector<std::vector<std::vector<double> > > weights, int channels, int stride, double bias) {

    std::vector<std::vector<std::vector<double> > > inputVol = NetUtil::arrayToVolume(input, channels);
    std::vector<std::vector<double> > output;

    int paddedLength = inputVol[0].size() + zP * 2;
    int fsSpread = floor(weights[0].size() / 2);
    int outSize = (inputVol[0].size() - weights[0].size() + 2*zP) / stride + 1;

    // Fill with 0 values
    for (int r=0; r<outSize; r++) {
        std::vector<double> row;
        for (int v=0; v<outSize; v++) {
            row.push_back(0.0);
        }
        output.push_back(row);
    }

    // For each input channel
    for (int ci=0; ci<channels; ci++) {

        inputVol[ci] = addZeroPadding(inputVol[ci], zP);

        // For each inputY without zP
        for (int inputY=fsSpread; inputY<paddedLength-fsSpread; inputY+=stride) {

            int vi = (inputY-fsSpread)/stride;

            // For each inputX without zP
            for (int inputX=fsSpread; inputX<paddedLength-fsSpread; inputX+=stride) {

                double sum = 0;

                // For each weightsY on input
                for (int weightsY=0; weightsY<weights[0].size(); weightsY++) {
                    // For each weightsX on input
                    for (int weightsX=0; weightsX<weights[0].size(); weightsX++) {
                        sum += inputVol[ci][inputY+(weightsY-fsSpread)][inputX+(weightsX-fsSpread)] * weights[ci][weightsY][weightsX];
                    }
                }

                output[vi][(inputX-fsSpread)/stride] += sum;
            }
        }
    }

    // Then add bias
    for (int outY=0; outY<output.size(); outY++) {
        for (int outX=0; outX<output.size(); outX++) {
            output[outY][outX] += bias;
        }
    }

    return output;
}

template <class T>
std::vector<std::vector<std::vector<T> > > NetUtil::createVolume (int depth, int rows, int columns, T value) {

    std::vector<std::vector<std::vector<T> > > volume;

    for (int d=0; d<depth; d++) {
        std::vector<std::vector<T> > map;

        for (int r=0; r<rows; r++) {
            std::vector<T> row;

            for (int c=0; c<columns; c++) {
                row.push_back(value);
            }

            map.push_back(row);
        }

        volume.push_back(map);
    }

    return volume;
}

void NetUtil::buildConvErrorMap (ConvLayer* layer, Layer* nextLayer, int filterI) {

    // Cache / convenience
    int zeroPadding = nextLayer->zeroPadding;
    int paddedLength = layer->filters[filterI]->errorMap.size() + zeroPadding*2;
    int fsSpread = floor(nextLayer->filterSize / 2);

    std::vector<std::vector<double> > errorMap;

    // Zero pad and clear the error map, to allow easy convoling
    for (int row=0; row<paddedLength; row++) {
        std::vector<double> paddedRow;

        for (int val=0; val<paddedLength; val++) {
            paddedRow.push_back(0);
        }

        errorMap.push_back(paddedRow);
    }

    // For each channel in filter in the next layer which corresponds to this filter
    for (int nlFilterI=0; nlFilterI<nextLayer->filters.size(); nlFilterI++) {

        std::vector<std::vector<double> > weights = nextLayer->filters[nlFilterI]->weights[filterI];
        std::vector<std::vector<double> > errMap = nextLayer->filters[nlFilterI]->errorMap;

        // Unconvolve their error map using the weights
        for (int inY=fsSpread; inY<paddedLength - fsSpread; inY+=nextLayer->stride) {
            for (int inX=fsSpread; inX<paddedLength - fsSpread; inX+=nextLayer->stride) {

                for (int wY=0; wY<nextLayer->filterSize; wY++) {
                    for (int wX=0; wX<nextLayer->filterSize; wX++) {
                        errorMap[inY+(wY-fsSpread)][inX+(wX-fsSpread)] += weights[wY][wX]
                            * errMap[(inY-fsSpread)/nextLayer->stride][(inX-fsSpread)/nextLayer->stride];
                    }
                }
            }
        }
    }

    // Take out the zero padding. Rows:
    errorMap.erase(errorMap.begin(), errorMap.begin()+zeroPadding);
    errorMap.erase(errorMap.end()-zeroPadding, errorMap.end());

    // Columns:
    for (int eY=0; eY<errorMap.size(); eY++) {
        errorMap[eY].erase(errorMap[eY].begin(), errorMap[eY].begin()+zeroPadding);
        errorMap[eY].erase(errorMap[eY].end()-zeroPadding, errorMap[eY].end());
    }

    layer->filters[filterI]->errorMap = errorMap;
}

void NetUtil::buildConvDWeights (ConvLayer* layer) {

    Network* net = Network::getInstance(layer->netInstance);
    int weightsCount = layer->filters[0]->weights[0].size();
    int fsSpread = floor(weightsCount / 2);
    int channelsCount = layer->filters[0]->weights.size();

    // Adding an intermediary step to allow regularization to work
    std::vector<std::vector<double> > deltaDeltaWeights;

    // Filling the deltaDeltaWeights with 0 values
    for (int weightsY=0; weightsY<weightsCount; weightsY++) {

        std::vector<double> deltaDeltaWeightsRow;

        for (int weightsX=0; weightsX<weightsCount; weightsX++) {
            deltaDeltaWeightsRow.push_back(0);
        }

        deltaDeltaWeights.push_back(deltaDeltaWeightsRow);
    }

    // For each filter
    for (int f=0; f<layer->filters.size(); f++) {

        // Each channel will take the error map and the corresponding inputMap from the input...
        for (int c=0; c<channelsCount; c++) {

            std::vector<double> inputValues = NetUtil::getActivations(layer->prevLayer, c, layer->inMapValuesCount);
            std::vector<std::vector<double> > inputMap = NetUtil::addZeroPadding(NetUtil::arrayToMap(inputValues, sqrt(layer->inMapValuesCount)), layer->zeroPadding);

            // ...slide the filter with correct stride across the zero-padded inputMap...
            for (int inY=fsSpread; inY<inputMap.size()-fsSpread; inY+= layer->stride) {
                for (int inX=fsSpread; inX<inputMap.size()-fsSpread; inX+= layer->stride) {

                    // ...and at each location...
                    for (int wY=0; wY<weightsCount; wY++) {
                        for (int wX=0; wX<weightsCount; wX++) {

                            double activation = inputMap[inY-fsSpread+wY][inX-fsSpread+wX];

                            // Increment and regularize the delta delta weights by the input activation (later multiplied by the error)
                            deltaDeltaWeights[wY][wX] += activation *
                                (1 + ((net->l2+net->l1)/net->miniBatchSize) * layer->filters[f]->weights[c][wY][wX]);
                        }
                    }

                    double error = layer->filters[f]->errorMap[(inY-fsSpread)/layer->stride][(inX-fsSpread)/layer->stride];

                    // Applying and resetting the deltaDeltaWeights
                    for (int wY=0; wY<weightsCount; wY++) {
                        for (int wX=0; wX<weightsCount; wX++) {
                            layer->filters[f]->deltaWeights[c][wY][wX] += deltaDeltaWeights[wY][wX] * error;
                            deltaDeltaWeights[wY][wX] = 0;
                        }
                    }
                }
            }
        }

        // Increment the deltaBias by the sum of all errors in the filter
        for (int eY=0; eY<layer->filters[f]->errorMap.size(); eY++) {
            for (int eX=0; eX<layer->filters[f]->errorMap.size(); eX++) {
                layer->filters[f]->deltaBias += layer->filters[f]->errorMap[eY][eX];
            }
        }
    }
}


std::vector<double> NetUtil::getActivations (Layer* layer) {

    std::vector<double> activations;

    if (layer->type == "FC") {
        for (int n=0; n<layer->size; n++) {
            activations.push_back(layer->neurons[n]->activation);
        }

    } else if (layer->type == "Conv") {

        for (int f=0; f<layer->filters.size(); f++) {
            for (int r=0; r<layer->filters[f]->activationMap.size(); r++) {
                for (int c=0; c<layer->filters[f]->activationMap[r].size(); c++) {
                    activations.push_back(layer->filters[f]->activationMap[r][c]);
                }
            }
        }
    }

    return activations;
}

std::vector<double> NetUtil::getActivations (Layer* layer, int mapStartI, int mapSize) {

    std::vector<double> activations;

    if (layer->type == "FC") {

        for (int n=mapStartI*mapSize; n<(mapStartI+1)*mapSize; n++) {
            activations.push_back(layer->neurons[n]->activation);
        }

    } else if (layer->type == "Conv") {

        for (int r=0; r<layer->filters[mapStartI]->activationMap.size(); r++) {
            for (int c=0; c<layer->filters[mapStartI]->activationMap[r].size(); c++) {
                activations.push_back(layer->filters[mapStartI]->activationMap[r][c]);
            }
        }
    }

    return activations;
}