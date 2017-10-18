
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