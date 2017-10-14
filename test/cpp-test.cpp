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

namespace Misc {
    TEST(standardDeviation, CalculatesValueCorrectly) {
        std::vector<double> vals = {3,5,7,8,5,25,8,4};
        EXPECT_EQ( standardDeviation(vals), 6.603739470936145 );
    }
}

namespace Network_cpp {

    // Appends a new instance to the Network::instances vector, returning instance index
    TEST(Network, newNetwork_1) {
        EXPECT_EQ( Network::netInstances.size(), 0 );
        Network::newNetwork();
        EXPECT_EQ( Network::netInstances.size(), 1 );
        Network::newNetwork();
        EXPECT_EQ( Network::netInstances.size(), 2 );
    }

    // Returns the index of the newly created instance
    TEST(Network, newNetwork_2) {
        Network::deleteNetwork();
        EXPECT_EQ( Network::newNetwork(), 0 );
        EXPECT_EQ( Network::newNetwork(), 1 );
        EXPECT_EQ( Network::newNetwork(), 2 );
    }

    // Returns the correct Network instance
    TEST(Network, getInstance) {
        Network::netInstances[0]->learningRate = (float) 1;
        Network::netInstances[1]->learningRate = (float) 2;
        Network::netInstances[2]->learningRate = (float) 3;
        EXPECT_EQ( Network::getInstance(0)->learningRate, (float) 1 );
        EXPECT_EQ( Network::getInstance(1)->learningRate, (float) 2 );
        EXPECT_EQ( Network::getInstance(2)->learningRate, (float) 3 );
    }

    // Deletes a network instance when given an index
    TEST(Network, deleteNetwork_1) {
        EXPECT_EQ( Network::netInstances.size(), 3 );
        EXPECT_FALSE( Network::netInstances[1] == 0 );
        Network::deleteNetwork(1);
        EXPECT_TRUE( Network::netInstances[1] == 0 );
    }

    // Deletes all network instances when no index is given
    TEST(Network, deleteNetwork_2) {
        EXPECT_EQ( Network::netInstances.size(), 3 );
        Network::deleteNetwork();
        EXPECT_EQ( Network::netInstances.size(), 0 );
    }

    // Assigns the network activation function to each layer
    TEST(Network, joinLayers_1) {
        Network::deleteNetwork();
        Network::newNetwork();
        Network::getInstance(0)->weightsConfig["limit"] = 0.1;
        Network::getInstance(0)->weightInitFn = &NetMath::uniform;
        Network::getInstance(0)->layers.push_back(new Layer(0, 3));
        Network::getInstance(0)->layers.push_back(new Layer(0, 3));
        Network::getInstance(0)->layers.push_back(new Layer(0, 3));
        Network::getInstance(0)->activation = &NetMath::sigmoid;

        EXPECT_NE( Network::getInstance(0)->layers[0]->activation, &NetMath::sigmoid );
        EXPECT_NE( Network::getInstance(0)->layers[1]->activation, &NetMath::sigmoid );
        Network::getInstance(0)->joinLayers();
        EXPECT_EQ( Network::getInstance(0)->layers[0]->activation, &NetMath::sigmoid );
        EXPECT_EQ( Network::getInstance(0)->layers[1]->activation, &NetMath::sigmoid );
    }

    // Assigns prevLayers to layers accordingly
    TEST(Network, joinLayers_2) {
        EXPECT_EQ( Network::getInstance(0)->layers[1]->prevLayer, Network::getInstance(0)->layers[0] );
        EXPECT_EQ( Network::getInstance(0)->layers[2]->prevLayer, Network::getInstance(0)->layers[1] );
    }

    // Assigns nextLayers to layers accordingly
    TEST(Network, joinLayers_3) {
        EXPECT_EQ( Network::getInstance(0)->layers[0]->nextLayer, Network::getInstance(0)->layers[1] );
        EXPECT_EQ( Network::getInstance(0)->layers[1]->nextLayer, Network::getInstance(0)->layers[2] );
    }

    // Assigns fanIn and fanOut correctly
    TEST(Network, joinLayers_4) {
        EXPECT_EQ( Network::getInstance(0)->layers[0]->fanIn,-1 );
        EXPECT_EQ( Network::getInstance(0)->layers[0]->fanOut, 3 );
        EXPECT_EQ( Network::getInstance(0)->layers[1]->fanIn, 3 );
        EXPECT_EQ( Network::getInstance(0)->layers[1]->fanOut, 3 );
        EXPECT_EQ( Network::getInstance(0)->layers[2]->fanIn, 3 );
        EXPECT_EQ( Network::getInstance(0)->layers[2]->fanOut,-1 );
    }

    // Sets the first layer's neurons' activations to the input given
    TEST(Network, forward_1) {
        std::vector<double> testInput = {1,2,3};

        EXPECT_NE( Network::getInstance(0)->layers[0]->neurons[0]->activation, 1 );
        EXPECT_NE( Network::getInstance(0)->layers[0]->neurons[1]->activation, 2 );
        EXPECT_NE( Network::getInstance(0)->layers[0]->neurons[2]->activation, 3 );

        Network::getInstance(0)->forward(testInput);

        EXPECT_EQ( Network::getInstance(0)->layers[0]->neurons[0]->activation, 1);
        EXPECT_EQ( Network::getInstance(0)->layers[0]->neurons[1]->activation, 2);
        EXPECT_EQ( Network::getInstance(0)->layers[0]->neurons[2]->activation, 3);
    }

    // Returns a vector of activations in the last layer
    TEST(Network, forward_2) {
        std::vector<double> testInput = {1,2,3};
        std::vector<double> returned = Network::getInstance(0)->forward(testInput);

        std::vector<double> actualValues;

        for (int i=0; i<3; i++) {
            actualValues.push_back(Network::getInstance(0)->layers[2]->neurons[i]->activation);
        }

        EXPECT_EQ( returned, actualValues );
    }

    // Sets all the delta weights values to 0
    TEST(Network, resetDeltaWeights) {
        Network::deleteNetwork();
        Network::newNetwork();
        Network::getInstance(0)->weightsConfig["limit"] = 0.1;
        Network::getInstance(0)->weightInitFn = &NetMath::uniform;
        Network::getInstance(0)->layers.push_back(new Layer(0, 3));
        Network::getInstance(0)->layers.push_back(new Layer(0, 3));
        Network::getInstance(0)->layers.push_back(new Layer(0, 3));
        Network::getInstance(0)->joinLayers();

        for (int n=1; n<3; n++) {
            Network::getInstance(0)->layers[0]->neurons[n]->deltaWeights = {1,1,1};
            Network::getInstance(0)->layers[1]->neurons[n]->deltaWeights = {1,1,1};
        }

        Network::getInstance(0)->resetDeltaWeights();
        std::vector<double> expected = {0,0,0};

        for (int n=1; n<3; n++) {
            EXPECT_EQ( Network::getInstance(0)->layers[1]->neurons[n]->deltaWeights, expected );
            EXPECT_EQ( Network::getInstance(0)->layers[2]->neurons[n]->deltaWeights, expected );
        }
    }

    // Increment the weights by the delta weights
    TEST(Network, applyDeltaWeights) {
        Network::getInstance(0)->learningRate = 1;
        Network::getInstance(0)->updateFnIndex = 0;
        for (int n=1; n<3; n++) {
            Network::getInstance(0)->layers[1]->neurons[n]->weights = {1,1,1};
            Network::getInstance(0)->layers[2]->neurons[n]->weights = {2,2,2};
            Network::getInstance(0)->layers[1]->neurons[n]->deltaWeights = {1,2,3};
            Network::getInstance(0)->layers[2]->neurons[n]->deltaWeights = {4,5,6};
        }

        std::vector<double> expected1 = {2,3,4};
        std::vector<double> expected2 = {6,7,8};

        Network::getInstance(0)->applyDeltaWeights();

        for (int n=1; n<3; n++) {
            EXPECT_EQ( Network::getInstance(0)->layers[1]->neurons[n]->weights, expected1 );
            EXPECT_EQ( Network::getInstance(0)->layers[2]->neurons[n]->weights, expected2 );
        }
    }
}


int main (int argc, char** argv) {
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}