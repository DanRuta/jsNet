
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

std::vector<std::vector<double> > NetUtil::convolve(std::vector<std::vector<std::vector<double> > > input, int zP,
    std::vector<std::vector<std::vector<double> > > weights, int channels, int stride, double bias) {

    std::vector<std::vector<double> > output;

    int outSize = (input[0].size() - weights[0].size() + 2*zP) / stride + 1;

    // Fill with 0 values
    for (int r=0; r<outSize; r++) {
        std::vector<double> row;
        for (int v=0; v<outSize; v++) {
            row.push_back(0.0);
        }
        output.push_back(row);
    }

    int x = -zP;
    int y = -zP;

    for (int outY=0; outY<outSize; y+=stride, outY++) {

        x = -zP;

        for (int outX=0; outX<outSize; x+=stride, outX++) {

            double sum = 0;

            for (int weightsY=0; weightsY<weights[0].size(); weightsY++) {

                int inputY = y+weightsY;

                for (int weightsX=0; weightsX<weights[0].size(); weightsX++) {

                    int inputX = x+weightsX;

                    if (inputY>=0 && inputY<input[0].size() && inputX>=0 && inputX<input[0].size()) {
                        for (int di=0; di<channels; di++) {
                            sum += input[di][inputY][inputX] * weights[di][weightsY][weightsX];
                        }
                    }
                }
            }

            sum += bias;

            output[outY][outX] = sum;
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

std::vector<std::vector<double> > NetUtil::buildConvErrorMap (int paddedLength, Layer* nextLayer, int filterI) {

    // Cache / convenience
    int zeroPadding = nextLayer->zeroPadding;
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

        std::vector<std::vector<double> > weights = nextLayer->filterWeights[nlFilterI][filterI];
        std::vector<std::vector<double> > errMap = nextLayer->errors[nlFilterI];

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

    return errorMap;
}

void NetUtil::buildConvDWeights (ConvLayer* layer) {

    int weightsCount = layer->filterWeights[0][0].size();
    int fsSpread = floor(weightsCount / 2);
    int channelsCount = layer->filterWeights[0].size();

    // For each filter
    for (int f=0; f<layer->filters.size(); f++) {

        // Each channel will take the error map and the corresponding inputMap from the input...
        for (int c=0; c<channelsCount; c++) {

            std::vector<double> inputValues = NetUtil::getActivations(layer->prevLayer, c, layer->inMapValuesCount);
            std::vector<std::vector<double> > inputMap = NetUtil::addZeroPadding(NetUtil::arrayToMap(inputValues, sqrt(layer->inMapValuesCount)), layer->zeroPadding);

            // ...slide the filter with correct stride across the zero-padded inputMap...
            for (int inY=fsSpread; inY<inputMap.size()-fsSpread; inY+= layer->stride) {
                for (int inX=fsSpread; inX<inputMap.size()-fsSpread; inX+= layer->stride) {

                    double error = layer->errors[f][(inY-fsSpread)/layer->stride][(inX-fsSpread)/layer->stride];

                    // ...and at each location...
                    for (int wY=0; wY<weightsCount; wY++) {
                        for (int wX=0; wX<weightsCount; wX++) {
                            // activation * error
                            layer->filterDeltaWeights[f][c][wY][wX] += inputMap[inY-fsSpread+wY][inX-fsSpread+wX] * error;
                        }
                    }
                }
            }
        }

        // Increment the deltaBias by the sum of all errors in the filter
        for (int eY=0; eY<layer->errors[f].size(); eY++) {
            for (int eX=0; eX<layer->errors[f].size(); eX++) {
                layer->deltaBiases[f] += layer->errors[f][eY][eX];
            }
        }
    }
}

std::vector<double> NetUtil::getActivations (Layer* layer, int mapStartI, int mapSize) {

    std::vector<double> activations;

    if (layer->type == "FC") {

        for (int n=mapStartI*mapSize; n<(mapStartI+1)*mapSize; n++) {
            activations.push_back(layer->actvns[n]);
        }

    } else if (layer->type == "Conv") {

        for (int r=0; r<layer->activations[mapStartI].size(); r++) {
            for (int c=0; c<layer->activations[mapStartI][r].size(); c++) {
                activations.push_back(layer->activations[mapStartI][r][c]);
            }
        }

    } else {

        for (int r=0; r<layer->activations[mapStartI].size(); r++) {
            for (int v=0; v<layer->activations[mapStartI].size(); v++) {
                activations.push_back(layer->activations[mapStartI][r][v]);
            }
        }

    }

    return activations;
}