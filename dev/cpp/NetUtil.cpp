
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

std::vector<std::vector<double> > NetUtil::convolve(std::vector<std::vector<std::vector<double> > > input,
     int zP, std::vector<std::vector<std::vector<double> > > weights, int channels, int stride, double bias) {

    std::vector<std::vector<double> > output;

    int paddedLength = input[0].size() + zP * 2;
    int fsSpread = floor(weights[0].size() / 2);
    int outSize = (input[0].size() - weights[0].size() + 2*zP) / stride + 1; // ??

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

        input[ci] = addZeroPadding(input[ci], zP);

        // For each inputY without zP
        for (int inputY=fsSpread; inputY<paddedLength-fsSpread; inputY+=stride) {

            int vi = (inputY-fsSpread)/stride;

            // For each inputX without zP
            for (int inputX=fsSpread; inputX<paddedLength-fsSpread; inputX+=stride) {

                int sum = 0;

                // For each weightsY on input
                for (int weightsY=0; weightsY<weights[0].size(); weightsY++) {
                    // For each weightsX on input
                    for (int weightsX=0; weightsX<weights[0].size(); weightsX++) {
                        sum += input[ci][inputY+(weightsY-fsSpread)][inputX+(weightsX-fsSpread)] * weights[ci][weightsY][weightsX];
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

std::vector<std::vector<std::vector<double> > > NetUtil::createVolume (int depth, int rows, int columns, int value) {

    std::vector<std::vector<std::vector<double> > > volume;

    for (int d=0; d<depth; d++) {
        std::vector<std::vector<double> > map;

        for (int r=0; r<rows; r++) {
            std::vector<double> row;

            for (int c=0; c<columns; c++) {
                row.push_back(value);
            }

            map.push_back(row);
        }

        volume.push_back(map);
    }

    return volume;
}
