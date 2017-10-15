#include "../dev/cpp/Network.cpp"
#include "cpp-mocks.cpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::MockFunction;
using ::testing::NiceMock;

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

MockLayer::MockLayer(int netI, int s) : Layer(netI, s) {
    netInstance = netI;
    size = s;
}
MockLayer::~MockLayer() {
    for (int n=0; n<neurons.size(); n++) {
        delete neurons[n];
    }
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
        Network::getInstance(0)->layers.push_back(new FCLayer(0, 3));
        Network::getInstance(0)->layers.push_back(new FCLayer(0, 3));
        Network::getInstance(0)->layers.push_back(new FCLayer(0, 3));
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

    class ForwardFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            l1 = new MockLayer(0, 3);
            l2 = new MockLayer(0, 2);
            l1->neurons.push_back(new Neuron());
            l1->neurons.push_back(new Neuron());
            l1->neurons.push_back(new Neuron());
            l2->neurons.push_back(new Neuron());
            l2->neurons.push_back(new Neuron());

            net->layers.push_back(l1);
            net->layers.push_back(l2);

            EXPECT_CALL(*l1, forward()).Times(0);
            EXPECT_CALL(*l2, forward()).Times(1);

            testInput = {1,2,3};
        }

        virtual void TearDown() {
            delete l1;
            delete l2;
            Network::deleteNetwork();
        }

        Network* net;
        MockLayer* l1;
        MockLayer* l2;
        std::vector<double> testInput;
    };

    // Calls the forward function of every layer except the first's
    TEST_F(ForwardFixture, forward_1) {
        MockLayer* l3 = new MockLayer(0, 1);
        net->layers.push_back(l3);
        l3->neurons.push_back(new Neuron());

        EXPECT_CALL(*l3, forward()).Times(1);

        net->forward(testInput);

        delete l3;
    }

    // Sets the first layer's neurons' activations to the input given
    TEST_F(ForwardFixture, forward_2) {

        EXPECT_NE( l1->neurons[0]->activation, 1 );
        EXPECT_NE( l1->neurons[1]->activation, 2 );
        EXPECT_NE( l1->neurons[2]->activation, 3 );

        net->forward(testInput);

        EXPECT_EQ( l1->neurons[0]->activation, 1);
        EXPECT_EQ( l1->neurons[1]->activation, 2);
        EXPECT_EQ( l1->neurons[2]->activation, 3);
    }


    // Returns a vector of activations in the last layer
    TEST_F(ForwardFixture, forward_3) {

        l2->neurons[0]->activation = 1;
        l2->neurons[1]->activation = 2;

        std::vector<double> returned = Network::getInstance(0)->forward(testInput);

        std::vector<double> actualValues = {1, 2};


        EXPECT_EQ( returned, actualValues );
    }

    // Calls the resetDeltaWeights function of each layer except the first's
    TEST(Network, resetDeltaWeights){
        Network::deleteNetwork();
        Network::newNetwork();
        MockLayer* l1 = new MockLayer(0, 3);
        MockLayer* l2 = new MockLayer(0, 3);
        MockLayer* l3 = new MockLayer(0, 3);
        Network::getInstance(0)->layers.push_back(l1);
        Network::getInstance(0)->layers.push_back(l2);
        Network::getInstance(0)->layers.push_back(l3);

        EXPECT_CALL(*l1, resetDeltaWeights()).Times(0);
        EXPECT_CALL(*l2, resetDeltaWeights()).Times(1);
        EXPECT_CALL(*l3, resetDeltaWeights()).Times(1);

        Network::getInstance(0)->resetDeltaWeights();

        delete l1;
        delete l2;
        delete l3;
    }

    // Calls the applyDeltaWeights function of each layer except the first's
    TEST(Network, applyDeltaWeights){
        Network::deleteNetwork();
        Network::newNetwork();
        MockLayer* l1 = new MockLayer(0, 3);
        MockLayer* l2 = new MockLayer(0, 3);
        MockLayer* l3 = new MockLayer(0, 3);
        Network::getInstance(0)->layers.push_back(l1);
        Network::getInstance(0)->layers.push_back(l2);
        Network::getInstance(0)->layers.push_back(l3);

        EXPECT_CALL(*l1, applyDeltaWeights()).Times(0);
        EXPECT_CALL(*l2, applyDeltaWeights()).Times(1);
        EXPECT_CALL(*l3, applyDeltaWeights()).Times(1);

        Network::getInstance(0)->applyDeltaWeights();

        delete l1;
        delete l2;
        delete l3;
    }
}

int main (int argc, char** argv) {
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}