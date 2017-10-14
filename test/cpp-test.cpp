#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../dev/cpp/Network.cpp"

double standardDeviation (std::vector<double> arr) {
    double avg = 0;

    for (int i=0; i<arr.size(); i++) {
        avg += arr[i];
    }

    avg /= arr.size();

    for (int i=0; i<arr.size(); i++) {
        arr[i] = pow(arr[i] - avg, 2);
    }

    double var = 0;

    for (int i=0; i<arr.size(); i++) {
        var += arr[i];
    }

    return sqrt(var / arr.size());
}

namespace {

    // Misc
    TEST(standardDeviation, Calculates_value_correctly) {
        std::vector<double> vals = {3,5,7,8,5,25,8,4};
        EXPECT_EQ( standardDeviation(vals), 6.603739470936145 );
    }

}

int main (int argc, char** argv) {
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}