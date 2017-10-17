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

namespace FCLayer_cpp {

    // Assigns the given layer pointer to this layer's nextLayer
    TEST(FCLayer, assignNext) {
        FCLayer* l1 = new FCLayer(0, 1);
        FCLayer* l2 = new FCLayer(0, 1);

        l1->assignNext(l2);
        EXPECT_EQ(l1->nextLayer, l2);

        delete l1;
        delete l2;
    }

    // Assigns the given layer pointer to this layer's prevLayer
    TEST(FCLayer, assignPrev) {
        FCLayer* l1 = new FCLayer(0, 1);
        FCLayer* l2 = new FCLayer(0, 1);

        l2->assignPrev(l1);
        EXPECT_EQ(l2->prevLayer, l1);

        delete l1;
        delete l2;
    }

    class InitFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->weightInitFn = &NetMath::uniform;
            l1 = new FCLayer(0, 2);
            l2 = new FCLayer(0, 5);
            net->layers.push_back(l1);
            net->layers.push_back(l2);
            l2->prevLayer = l1;
            l2->netInstance = 0;
        }

        virtual void TearDown() {
            delete l1;
            delete l2;
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* l1;
        FCLayer* l2;
    };

    // Fills the layers' neurons vectors with as many neurons as the layers' sizes
    TEST_F(InitFixture, init_1) {
        EXPECT_NE( l2->neurons.size(), 5 );
        l2->init(1);
        EXPECT_EQ( l2->neurons.size(), 5 );
    }

    // Sets the bias of every neuron to a number between -0.1 and 0.1
    TEST_F(InitFixture, init_2) {
        l2->init(1);

        for (int n=0; n<5; n++) {
            EXPECT_GE( l2->neurons[n]->bias, -0.1 );
            EXPECT_LE( l2->neurons[n]->bias, 0.1 );
        }
    }

    // Inits the neurons' weights vector with as many weights as there are neurons in the prev layer. (none in first layer)
    TEST_F(InitFixture, init_3) {
        l1->init(0);
        l2->init(1);

        EXPECT_EQ( l1->neurons[0]->weights.size(), 0 );
        EXPECT_EQ( l1->neurons[1]->weights.size(), 0 );

        for (int n=0; n<5; n++) {
            EXPECT_EQ( l2->neurons[n]->weights.size(), 2 );
        }
    }


    class FCForwardFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            Network::newNetwork();
            net = Network::getInstance(1);
            net->weightInitFn = &NetMath::uniform;
            l1 = new FCLayer(1, 2);
            l2 = new FCLayer(1, 5);
            net->layers.push_back(l1);
            net->layers.push_back(l2);
            l2->prevLayer = l1;
            l2->netInstance = 1;
            l1->init(0);
            l2->init(1);
            l2->activation = &NetMath::sigmoid;
            l1->neurons[0]->activation = 1;
            l1->neurons[1]->activation = 2;
        }

        virtual void TearDown() {
            delete l1;
            delete l2;
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* l1;
        FCLayer* l2;
    };

    // SETS the neurons' sum to their bias + weighted activations of last layer's neurons
    TEST_F(FCForwardFixture, forward_1) {

        for (int n=0; n<3; n++) {
            l2->neurons[n]->weights = {1,2};
            l2->neurons[n]->bias = n;
        }

        net->isTraining= false;
        net->dropout = 1;

        l2->forward();

        EXPECT_EQ( l2->neurons[0]->sum, 5);
        EXPECT_EQ( l2->neurons[1]->sum, 6);
        EXPECT_EQ( l2->neurons[2]->sum, 7);

        // Check that it SETS it, and doesn't increment it
        l2->forward();

        EXPECT_FALSE( net->isTraining );
        EXPECT_EQ( l2->neurons[0]->sum, 5);
        EXPECT_EQ( l2->neurons[1]->sum, 6);
        EXPECT_EQ( l2->neurons[2]->sum, 7);

    }

    // Sets the layer's neurons' activation to the result of the activation function
    TEST_F(FCForwardFixture, forward_2) {

        for (int n=0; n<3; n++) {
            l2->neurons[n]->weights = {1,2};
            l2->neurons[n]->bias = n;
        }

        net->isTraining = false;
        net->dropout = 1;
        l2->forward();

        EXPECT_FALSE( net->isTraining );
        EXPECT_DOUBLE_EQ( l2->neurons[0]->activation, 0.9933071490757153 );
        EXPECT_DOUBLE_EQ( l2->neurons[1]->activation, 0.9975273768433653 );
        EXPECT_DOUBLE_EQ( l2->neurons[2]->activation, 0.9990889488055994 );
    }

    // Sets the neurons' dropped value to true and activation to 0 if the net is training and dropout is set to 0
    TEST_F(FCForwardFixture, forward_3) {

        for (int n=0; n<3; n++) {
            l2->neurons[n]->weights = {1,2};
            l2->neurons[n]->bias = n;
            l2->neurons[n]->sum = 0;
        }

        net->dropout = 0;
        net->isTraining = true;
        l2->forward();

        EXPECT_TRUE( l2->neurons[0]->dropped );
        EXPECT_TRUE( l2->neurons[1]->dropped );
        EXPECT_TRUE( l2->neurons[2]->dropped );

        EXPECT_EQ( l2->neurons[0]->activation, 0 );
        EXPECT_EQ( l2->neurons[1]->activation, 0 );
        EXPECT_EQ( l2->neurons[2]->activation, 0 );

        EXPECT_EQ( l2->neurons[0]->sum, 0 );
        EXPECT_EQ( l2->neurons[1]->sum, 0 );
        EXPECT_EQ( l2->neurons[2]->sum, 0 );
    }

    // Does not set neurons to dropped if the net is not training
    TEST_F(FCForwardFixture, forward_4) {
        for (int n=0; n<3; n++) {
            l2->neurons[n]->weights = {1,2};
            l2->neurons[n]->bias = n;
            l2->neurons[n]->activation = 0;
        }

        net->dropout = 1;
        net->isTraining = false;
        l2->forward();

        EXPECT_FALSE( l2->neurons[0]->dropped );
        EXPECT_FALSE( l2->neurons[1]->dropped );
        EXPECT_FALSE( l2->neurons[2]->dropped );

        EXPECT_DOUBLE_EQ( l2->neurons[0]->activation, 0.9933071490757153 );
        EXPECT_DOUBLE_EQ( l2->neurons[1]->activation, 0.9975273768433653 );
        EXPECT_DOUBLE_EQ( l2->neurons[2]->activation, 0.9990889488055994 );
    }

    // Divides the activation values by the dropout
    TEST_F(FCForwardFixture, forward_5) {
        for (int n=0; n<3; n++) {
            l2->neurons[n]->weights = {1,2};
            l2->neurons[n]->bias = n;
        }

        net->dropout = 0.5;
        net->isTraining = false;
        l2->forward();

        EXPECT_FALSE( net->isTraining );
        EXPECT_DOUBLE_EQ( l2->neurons[0]->activation, 0.9933071490757153 * 2 );
        EXPECT_DOUBLE_EQ( l2->neurons[1]->activation, 0.9975273768433653 * 2 );
        EXPECT_DOUBLE_EQ( l2->neurons[2]->activation, 0.9990889488055994 * 2 );
    }

    class FCBackwardFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->weightInitFn = &NetMath::uniform;
            l1 = new FCLayer(0, 2);
            l2 = new FCLayer(0, 3);
            l3 = new FCLayer(0, 4);
            l2->prevLayer = l1;
            l2->netInstance = 0;
            l2->netInstance = 0;
            l3->prevLayer = l2;
            l2->nextLayer = l3;
            l1->init(0);
            l2->init(1);
            l3->init(2);
            l2->activation = &NetMath::sigmoid;
        }

        virtual void TearDown() {
            delete l1;
            delete l2;
            delete l3;
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* l1;
        FCLayer* l2;
        FCLayer* l3;
    };

    // Sets the neurons' errors to difference between their activations and expected values, when provided
    TEST_F(FCBackwardFixture, backward_1) {
        std::vector<double> expected = {1,2,3};

        l2->neurons[0]->activation = 0;
        l2->neurons[1]->activation = 1;
        l2->neurons[2]->activation = 0;

        l2->neurons[0]->dropped = false;
        l2->neurons[1]->dropped = false;
        l2->neurons[2]->dropped = false;

        l2->backward(expected);

        EXPECT_EQ( l2->neurons[0]->error, 1 );
        EXPECT_EQ( l2->neurons[1]->error, 1 );
        EXPECT_EQ( l2->neurons[2]->error, 3 );
    }

    // Sets the neurons' derivatives to the activation prime of their sum, when no expected data is passed
    TEST_F(FCBackwardFixture, backward_2) {
        std::vector<double> emptyVec;

        l2->neurons[0]->sum = 0;
        l2->neurons[1]->sum = 1;
        l2->neurons[2]->sum = 0;
        l2->neurons[0]->dropped = false;
        l2->neurons[1]->dropped = false;
        l2->neurons[2]->dropped = false;
        l2->backward(emptyVec);

        EXPECT_EQ( l2->neurons[0]->derivative, 0.25 );
        EXPECT_DOUBLE_EQ( l2->neurons[1]->derivative, 0.19661193324148185 );
        EXPECT_EQ( l2->neurons[2]->derivative, 0.25 );
    }

    // Sets the neurons' errors to their derivative multiplied by weighted errors in next layer, when no expected data is passed
    TEST_F(FCBackwardFixture, backward_3) {
        std::vector<double> emptyVec;

        l2->neurons[0]->sum = 0.5;
        l2->neurons[1]->sum = 0.5;
        l2->neurons[2]->sum = 0.5;
        l2->neurons[0]->dropped = false;
        l2->neurons[1]->dropped = false;
        l2->neurons[2]->dropped = false;

        for (int i=0; i<4; i++) {
            l3->neurons[i]->error = 0.5;
            l3->neurons[i]->weights = {1,1,1,1};
        }

        l2->backward(emptyVec);

        EXPECT_DOUBLE_EQ( l2->neurons[0]->error, 0.470007424403189 );
        EXPECT_DOUBLE_EQ( l2->neurons[1]->error, 0.470007424403189 );
        EXPECT_DOUBLE_EQ( l2->neurons[2]->error, 0.470007424403189 );
    }

    // Increments each of its delta weights by its error * the respective weight's neuron's activation
    TEST_F(FCBackwardFixture, backward_4) {
        std::vector<double> expected = {1,2,3,4};

        Network::getInstance(0)->l2 = 0;

        l2->neurons[0]->activation = 0.5;
        l2->neurons[1]->activation = 0.5;
        l2->neurons[2]->activation = 0.5;
        l3->neurons[0]->dropped = false;
        l3->neurons[1]->dropped = false;
        l3->neurons[2]->dropped = false;
        l3->neurons[3]->dropped = false;

        for (int i=0; i<4; i++) {
            l3->neurons[i]->activation = 0.5;
        }

        l3->backward(expected);

        for (int n=0; n<4; n++) {
            EXPECT_EQ( l3->neurons[n]->deltaWeights[0], 0.25 + n * 0.5 );
            EXPECT_EQ( l3->neurons[n]->deltaWeights[1], 0.25 + n * 0.5 );
            EXPECT_EQ( l3->neurons[n]->deltaWeights[2], 0.25 + n * 0.5 );
        }
    }

    // Sets the neurons' deltaBias to their errors
    TEST_F(FCBackwardFixture, backward_5) {
        std::vector<double> expected = {1,2,3};

        l2->neurons[0]->activation = 0;
        l2->neurons[1]->activation = 1;
        l2->neurons[2]->activation = 0;
        l2->neurons[0]->dropped = false;
        l2->neurons[1]->dropped = false;
        l2->neurons[2]->dropped = false;

        l2->backward(expected);

        EXPECT_EQ( l2->neurons[0]->error, 1 );
        EXPECT_EQ( l2->neurons[0]->deltaBias, 1 );
        EXPECT_EQ( l2->neurons[1]->error, 1 );
        EXPECT_EQ( l2->neurons[1]->deltaBias, 1 );
        EXPECT_EQ( l2->neurons[2]->error, 3 );
        EXPECT_EQ( l2->neurons[2]->deltaBias, 3 );
    }

    // Sets the neurons' error and deltaBias values to 0 when they are dropped
    TEST_F(FCBackwardFixture, backward_6) {
        std::vector<double> expected = {1,2,3};

        l2->neurons[0]->activation = 0;
        l2->neurons[1]->activation = 1;
        l2->neurons[2]->activation = 0;

        l2->neurons[0]->dropped = true;
        l2->neurons[1]->dropped = true;
        l2->neurons[2]->dropped = true;
        l2->neurons[0]->error = 123;
        l2->neurons[1]->error = 123;
        l2->neurons[2]->error = 123;
        l2->neurons[0]->deltaBias = 456;
        l2->neurons[2]->deltaBias = 456;
        l2->neurons[1]->deltaBias = 456;

        l2->backward(expected);

        EXPECT_EQ( l2->neurons[0]->error, 0 );
        EXPECT_EQ( l2->neurons[1]->error, 0 );
        EXPECT_EQ( l2->neurons[2]->error, 0 );

        EXPECT_EQ( l2->neurons[0]->deltaBias, 0 );
        EXPECT_EQ( l2->neurons[1]->deltaBias, 0 );
        EXPECT_EQ( l2->neurons[2]->deltaBias, 0 );
    }

    // Increments the deltaWeights by the orig value, multiplied by the l2 amount * existing deltaWeight value
    TEST_F(FCBackwardFixture, backward_7) {
        std::vector<double> expected = {0.3, 0.3, 0.3, 0.3};
        Network::getInstance(0)->l2 = 0.001;

        l2->neurons[0]->activation = 0.5;
        l2->neurons[1]->activation = 0.5;
        l2->neurons[2]->activation = 0.5;
        l3->neurons[0]->dropped = false;
        l3->neurons[1]->dropped = false;
        l3->neurons[2]->dropped = false;
        l3->neurons[3]->dropped = false;

        for (int i=0; i<4; i++) {
            l3->neurons[i]->activation = 0.25;
            l3->neurons[i]->deltaWeights[0] = 0.25;
            l3->neurons[i]->deltaWeights[1] = 0.25;
            l3->neurons[i]->deltaWeights[2] = 0.25;
        }

        l3->backward(expected);

        for (int n=0; n<4; n++) {
            EXPECT_NEAR( l3->neurons[n]->deltaWeights[0], 0.27500625, 1e-6 );
            EXPECT_NEAR( l3->neurons[n]->deltaWeights[1], 0.27500625, 1e-6 );
            EXPECT_NEAR( l3->neurons[n]->deltaWeights[2], 0.27500625, 1e-6 );
        }
    }

    // Increments the deltaWeights by the orig value, multiplied by the l1 amount * existing deltaWeight value
    TEST_F(FCBackwardFixture, backward_8) {
        std::vector<double> expected = {0.3, 0.3, 0.3, 0.3};
        Network::getInstance(0)->l1 = 0.005;

        l2->neurons[0]->activation = 0.5;
        l2->neurons[1]->activation = 0.5;
        l2->neurons[2]->activation = 0.5;
        l3->neurons[0]->dropped = false;
        l3->neurons[1]->dropped = false;
        l3->neurons[2]->dropped = false;
        l3->neurons[3]->dropped = false;

        for (int i=0; i<4; i++) {
            l3->neurons[i]->activation = 0.25;
            l3->neurons[i]->deltaWeights = {0.25, 0.25, 0.25, 0.25};
        }

        l3->backward(expected);

        for (int n=0; n<4; n++) {
            EXPECT_NEAR( l3->neurons[n]->deltaWeights[0], 0.275031, 1e-4 );
            EXPECT_NEAR( l3->neurons[n]->deltaWeights[1], 0.275031, 1e-4 );
            EXPECT_NEAR( l3->neurons[n]->deltaWeights[2], 0.275031, 1e-4 );
        }
    }

    class FCApplyDeltaWeightsFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            l1 = new FCLayer(0, 2);
            l2 = new FCLayer(0, 3);
            l3 = new FCLayer(0, 1);
            l4 = new FCLayer(0, 1);
            Network* net = Network::getInstance(0);
            net->l2 = 0.001;
            net->l2Error = 0;
            net->l1 = 0.005;
            net->updateFnIndex = 0;
            net->learningRate = 1;
            net->layers.push_back(l1);
            net->layers.push_back(l2);
            net->layers.push_back(l3);
            net->layers.push_back(l4);
            l2->prevLayer = l1;
            l3->prevLayer = l2;
            l4->prevLayer = l3;
            l2->netInstance = 0;

            l2->neurons.push_back(new Neuron());
            l2->neurons.push_back(new Neuron());
            l2->neurons.push_back(new Neuron());

            l3->neurons.push_back(new Neuron());
            l4->neurons.push_back(new Neuron());
        }

        virtual void TearDown() {
            delete l1;
            delete l2;
            delete l3;
            delete l4;
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* l1;
        FCLayer* l2;
        FCLayer* l3;
        FCLayer* l4;
    };

    // Increments the weights by the delta weights
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_1) {

        for (int n=1; n<3; n++) {
            l2->neurons[n]->weights = {1,1,1};
            l2->neurons[n]->deltaWeights = {1,2,3};
        }

        l2->applyDeltaWeights();
        std::vector<double> expected = {2,3,4};

        for (int n=1; n<3; n++) {
            EXPECT_EQ( l2->neurons[n]->weights, expected );
        }
    }

    // Increments the bias by the deltaBias
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_2) {
        for (int n=0; n<3; n++) {
            l2->neurons[n]->bias = n;
            l2->neurons[n]->deltaBias = n*2;
        }

        l2->applyDeltaWeights();

        EXPECT_EQ( l2->neurons[0]->bias, 0 );
        EXPECT_EQ( l2->neurons[1]->bias, 3 );
        EXPECT_EQ( l2->neurons[2]->bias, 6 );
    }

    // Increments the l2Error by the l2 formula applied to all weights
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_3) {
        l4->neurons[0]->weights = {0.25, 0.25};
        l4->neurons[0]->deltaWeights = {0.5, 0.5};

        l4->applyDeltaWeights();

        EXPECT_NEAR( Network::getInstance(l4->netInstance)->l2Error, 0.0000625 , 1e-6 );
    }

    // Increments the l1Error by the l1 formula applied to all weights
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_4) {
        l4->neurons[0]->weights = {0.25, 0.25};
        l4->neurons[0]->deltaWeights = {0.5, 0.5};

        l4->applyDeltaWeights();

        EXPECT_NEAR( Network::getInstance(l4->netInstance)->l1Error, 0.0025 , 1e-6 );
    }

    // Sets all deltaWeight values to 0
    TEST(FCLayer, resetDeltaWeights) {
        Network::deleteNetwork();
        Network::newNetwork();
        FCLayer* l1 = new FCLayer(0, 2);
        FCLayer* l2 = new FCLayer(0, 3);
        l2->assignPrev(l1);
        l2->prevLayer = l1;
        l2->neurons.push_back(new Neuron());
        l2->neurons.push_back(new Neuron());
        l2->neurons.push_back(new Neuron());

        for (int n=1; n<3; n++) {
            l2->neurons[n]->deltaWeights = {1,2,3};
        }
        std::vector<double> expected = {0,0,0};

        l2->resetDeltaWeights();

        for (int n=1; n<3; n++) {
            EXPECT_EQ(l2->neurons[n]->deltaWeights, expected);
        }

        delete l1;
        delete l2;
        Network::deleteNetwork();
    }
}


int main (int argc, char** argv) {
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}