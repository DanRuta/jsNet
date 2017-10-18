
void NetUtil::shuffle (std::vector<std::tuple<std::vector<double>, std::vector<double> > > &values) {

    for (int i=values.size(); i; i--) {
        int j = floor(rand() / RAND_MAX * i);
        std::tuple<std::vector<double>, std::vector<double> > x = values[i-1];
        values[i-1] = values[j];
        values[j] = x;
    }
}