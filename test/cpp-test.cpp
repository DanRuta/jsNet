#include "../dev/cpp/Network.cpp"
#include "cpp-mocks.cpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::MockFunction;

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
            Network::newNetwork();
            net = Network::getInstance(1);
            net->weightInitFn = &NetMath::uniform;
            net->miniBatchSize = 1;
            l1 = new FCLayer(1, 2);
            l2 = new FCLayer(1, 3);
            l3 = new FCLayer(1, 4);
            l2->prevLayer = l1;
            l2->netInstance = 1;
            l2->netInstance = 1;
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
        net->l2 = 0.001;

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

        EXPECT_EQ( net->l2, 0.001 );

        for (int n=0; n<4; n++) {
            EXPECT_NEAR( l3->neurons[n]->deltaWeights[0], 0.27500625, 1e-3 );
            EXPECT_NEAR( l3->neurons[n]->deltaWeights[1], 0.27500625, 1e-3 );
            EXPECT_NEAR( l3->neurons[n]->deltaWeights[2], 0.27500625, 1e-3 );
        }
    }

    // Increments the deltaWeights by the orig value, multiplied by the l1 amount * existing deltaWeight value
    TEST_F(FCBackwardFixture, backward_8) {
        std::vector<double> expected = {0.3, 0.3, 0.3, 0.3};
        net->l1 = 0.005;

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

    // Regularizes by a tenth as much when the miniBatchSize is configured as 10
    TEST_F(FCBackwardFixture, backward_9) {
        std::vector<double> expected = {0.3, 0.3, 0.3, 0.3};
        l2->neurons[0]->activation = 0.5;
        l2->neurons[1]->activation = 0.5;
        l2->neurons[2]->activation = 0.5;
        l3->neurons[0]->deltaWeights = {0.25, 0.25, 0.25, 0.25};
        l3->neurons[0]->activation = 0.25;
        l3->neurons[1]->deltaWeights = {0.25, 0.25, 0.25, 0.25};
        l3->neurons[1]->activation = 0.25;
        l3->neurons[2]->deltaWeights = {0.25, 0.25, 0.25, 0.25};
        l3->neurons[2]->activation = 0.25;
        l3->neurons[3]->deltaWeights = {0.25, 0.25, 0.25, 0.25};
        l3->neurons[3]->activation = 0.25;
        net->l1 = 0.005;
        net->miniBatchSize = 10;

        l3->backward(expected);

        EXPECT_NEAR(l3->neurons[0]->deltaWeights[0], 0.275003, 1e-6);

        l3->neurons[0]->deltaWeights = {0.25, 0.25, 0.25, 0.25};
        l3->neurons[0]->activation = 0.25;
        l3->neurons[1]->deltaWeights = {0.25, 0.25, 0.25, 0.25};
        l3->neurons[1]->activation = 0.25;
        l3->neurons[2]->deltaWeights = {0.25, 0.25, 0.25, 0.25};
        l3->neurons[2]->activation = 0.25;
        l3->neurons[3]->deltaWeights = {0.25, 0.25, 0.25, 0.25};
        l3->neurons[3]->activation = 0.25;
        net->l2 = 0.001;
        net->l1 = 0;

        l3->backward(expected);

        EXPECT_NEAR(l3->neurons[0]->deltaWeights[0], 0.275001, 1e-6);
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


    // ====== This test seems to fail only when Travis runs it ======
    // Increments the weights by the delta weights
    // TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_1) {

    //     l2->neurons[0]->weights = {1,1,1};
    //     l2->neurons[0]->deltaWeights = {1,2,3};
    //     l2->neurons[1]->weights = {1,1,1};
    //     l2->neurons[1]->deltaWeights = {1,2,3};

    //     l2->applyDeltaWeights();
    //     std::vector<double> expected = {2,3,4};

    //     EXPECT_EQ( l2->neurons[0]->weights, expected );
    //     EXPECT_EQ( l2->neurons[1]->weights, expected );
    // }

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


namespace Neuron_cpp {

    class NeuronInitFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            testN = new Neuron();
        }

        virtual void TearDown() {
            delete testN;
            Network::deleteNetwork();
        }

        Network* net;
        Neuron* testN;
    };

    // Fills the deltaWeights vector with 0 values, matching weights size
    TEST_F(NeuronInitFixture, init_1) {
        testN->weights = {1,2,3,4,5};
        testN->init(0);
        EXPECT_EQ( testN->deltaWeights.size(), 5 );
    }

    // Sets the neuron deltaBias to 0
    TEST_F(NeuronInitFixture, init_2) {
        testN->deltaBias = 999;
        testN->init(0);
        EXPECT_EQ( testN->deltaBias, 0 );
    }

    // Sets the neuron biasGain to 1 if the net's updateFn is gain
    TEST_F(NeuronInitFixture, init_3) {
        // testN->biasGain = 99;
        net->updateFnIndex = 1;
        testN->init(0);
        EXPECT_EQ( testN->biasGain, 1 );
    }

    // Sets the neuron weightGain to a vector of 1s, with the same size as the weights vector when updateFn is gain
    TEST_F(NeuronInitFixture, init_4) {
        testN->weights = {1,2,3,4,5};
        net->updateFnIndex = 1;
        testN->init(0);
        std::vector<double> expected = {1,1,1,1,1};
        EXPECT_EQ( testN->weightGain, expected );
    }

    // Does not set the biasGain or weightGain to anything if updateFn is not gain
    TEST_F(NeuronInitFixture, init_5) {
        net->updateFnIndex = 2;
        testN->init(0);
        EXPECT_EQ( testN->weightGain.size(), 0 );
    }

    // Sets the neuron biasCache to 0 if the updateFn is adagrad
    TEST_F(NeuronInitFixture, init_6) {
        net->updateFnIndex = 2;
        testN->biasCache = 1;
        testN->init(0);
        EXPECT_EQ( testN->biasCache, 0 );
    }

    // Sets the neuron weightsCache to a vector of zeroes with the same size as the weights when updateFn is adagrad
    TEST_F(NeuronInitFixture, init_7) {
        testN->weights = {1,2,3,4,5};
        net->updateFnIndex = 2;
        testN->init(0);
        std::vector<double> expected = {0,0,0,0,0};
        EXPECT_EQ( testN->weightsCache, expected );
    }

    // Does not set the biasCache or weightsCache to anything if updateFn is not adagrad
    TEST_F(NeuronInitFixture, init_8) {
        net->updateFnIndex = 1;
        testN->biasCache = 12234;
        testN->init(0);
        EXPECT_EQ( testN->biasCache, 12234 );
        EXPECT_EQ( testN->weightsCache.size(), 0 );
    }

    // Sets the neuron biasCache to 0 if the updateFn is rmsprop
    TEST_F(NeuronInitFixture, init_9) {
        net->updateFnIndex = 3;
        testN->biasCache = 1;
        testN->init(0);
        EXPECT_EQ( testN->biasCache, 0 );
    }

    // Sets the neuron weightsCache to a vector of zeroes with the same size as the weights when updateFn is rmsprop
    TEST_F(NeuronInitFixture, init_10) {
        testN->weights = {1,2,3,4,5};
        net->updateFnIndex = 3;
        testN->init(0);
        std::vector<double> expected = {0,0,0,0,0};
        EXPECT_EQ( testN->weightsCache, expected );
    }

    // Sets the neuron m and neuron v to 0 if the updateFn is adam
    TEST_F(NeuronInitFixture, init_11) {
        net->updateFnIndex = 4;
        testN->m = 1;
        testN->v = 1;
        testN->init(0);
        EXPECT_EQ( testN->m, 0 );
        EXPECT_EQ( testN->v, 0 );
    }

    // Does not set the neuron m and neuron v to 0 if the updateFn is not adam
    TEST_F(NeuronInitFixture, init_12) {
        net->updateFnIndex = 3;
        testN->m = 1;
        testN->v = 1;
        testN->init(0);
        EXPECT_EQ( testN->m, 1 );
        EXPECT_EQ( testN->v, 1 );
    }

    // Sets the neuron biasCache and adadeltaBiasCache to 0 if the updateFn is adadelta
    TEST_F(NeuronInitFixture, init_13) {
        net->updateFnIndex = 5;
        testN->biasCache = 1;
        testN->adadeltaBiasCache = 1;
        testN->init(0);
        EXPECT_EQ( testN->biasCache, 0 );
        EXPECT_EQ( testN->adadeltaBiasCache, 0 );
    }

    // Sets the neuron weightsCache and adadeltaCache to a vector of zeroes with the same size as the weights when updateFn is adadelta
    TEST_F(NeuronInitFixture, init_14) {
        testN->weights = {1,2,3,4,5};
        net->updateFnIndex = 5;
        testN->init(0);
        std::vector<double> expected = {0,0,0,0,0};
        EXPECT_EQ( testN->weightsCache, expected );
        EXPECT_EQ( testN->adadeltaCache, expected );
    }

    // Does not set the biasCache or weightsCache to anything if updateFn is not adadelta
    TEST_F(NeuronInitFixture, init_15) {
        net->updateFnIndex = 1;
        testN->biasCache = 12234;
        testN->adadeltaBiasCache = 12234;
        testN->init(0);
        EXPECT_EQ( testN->biasCache, 12234 );
        EXPECT_EQ( testN->adadeltaBiasCache, 12234 );
        EXPECT_EQ( testN->weightsCache.size(), 0 );
        EXPECT_EQ( testN->adadeltaCache.size(), 0 );
    }

    // Sets the neuron rreluSlope to a number if the activation is rrelu
    TEST_F(NeuronInitFixture, init_16) {
        net->activation = &NetMath::rrelu;
        testN->rreluSlope = 0.1;
        testN->init(0);
        EXPECT_NE( testN->rreluSlope, 0 );
        EXPECT_NE( testN->rreluSlope, 0.1 );
        EXPECT_GE( testN->rreluSlope, -0.1);
        EXPECT_LE( testN->rreluSlope, 0.1);
    }

    // Sets the network eluAlpha to the neuron, if the activation function is elu
    TEST_F(NeuronInitFixture, init_17) {
        net->activation = &NetMath::elu;
        net->eluAlpha = 0.1;
        testN->init(0);
        EXPECT_NEAR(testN->eluAlpha, 0.1, 1e-6 );
    }
}


namespace NetMath_cpp {

    TEST(NetMath, sigmoid) {
        Neuron* testN = new Neuron();
        EXPECT_EQ( NetMath::sigmoid(1.681241237, false, testN), 0.8430688214048092 );
        EXPECT_EQ( NetMath::sigmoid(0.8430688214048092, true, testN), 0.21035474941074114 );
        delete testN;
    }

    TEST(NetMath, tanh) {
        Neuron* testN = new Neuron();
        EXPECT_EQ( NetMath::tanh(1, false, testN), 0.7615941559557649 );
        EXPECT_EQ( NetMath::tanh(0.5, false, testN), 0.46211715726000974);
        EXPECT_EQ( NetMath::tanh(0.5, true, testN), 0.7864477329659275 );
        EXPECT_EQ( NetMath::tanh(1.5, true, testN), 0.18070663892364855 );
        EXPECT_NE( NetMath::tanh(900, true, testN), NAN );
        delete testN;
    }

    TEST(NetMath, lecuntanh) {
        Neuron* testN = new Neuron();
        EXPECT_EQ( NetMath::lecuntanh(2.0, false, testN), 1.4929388053842507 );
        EXPECT_EQ( NetMath::lecuntanh(-2.0, false, testN), -1.4929388053842507 );
        EXPECT_EQ( NetMath::lecuntanh(2.0, true, testN), 0.2802507761872869 );
        EXPECT_EQ( NetMath::lecuntanh(-2.0, true, testN), 0.2802507761872869 );
        delete testN;
    }

    TEST(NetMath, relu) {
        Neuron* testN = new Neuron();
        EXPECT_EQ( NetMath::relu(2, false, testN), 2 );
        EXPECT_EQ( NetMath::relu(-2, false, testN), 0 );
        EXPECT_EQ( NetMath::relu(2, true, testN), 1 );
        EXPECT_EQ( NetMath::relu(-2, true, testN), 0 );
        delete testN;
    }

    TEST(NetMath, lrelu) {
        Neuron* testN = new Neuron();
        testN->lreluSlope = -0.0005;
        EXPECT_EQ( NetMath::lrelu(2, false, testN), 2 );
        EXPECT_EQ( NetMath::lrelu(-2, false, testN), -0.001 );
        EXPECT_EQ( NetMath::lrelu(2, true, testN), 1 );
        EXPECT_EQ( NetMath::lrelu(-2, true, testN), -0.0005 );
        delete testN;
    }

    TEST(NetMath, rrelu) {
        Neuron* testN = new Neuron();
        testN->rreluSlope = 0.0005;
        EXPECT_EQ( NetMath::rrelu(2, false, testN), 2 );
        EXPECT_EQ( NetMath::rrelu(-2, false, testN), 0.0005 );
        EXPECT_EQ( NetMath::rrelu(2, true, testN), 1 );
        EXPECT_EQ( NetMath::rrelu(-2, true, testN), 0.0005 );
        delete testN;
    }

    TEST(NetMath, elu) {
        Neuron* testN = new Neuron();
        testN->eluAlpha = 1;
        EXPECT_EQ( NetMath::elu(2, false, testN), 2 );
        EXPECT_EQ( NetMath::elu(-0.25, false, testN), -0.22119921692859512 );
        EXPECT_EQ( NetMath::elu(2, true, testN), 1 );
        EXPECT_EQ( NetMath::elu(-0.5, true, testN), 0.6065306597126334 );
        delete testN;
    }

    TEST(NetMath, meansquarederror) {
        std::vector<double> values1 = {13,17,18,20,24};
        std::vector<double> values2 = {12,15,20,22,24};
        EXPECT_EQ( NetMath::meansquarederror(values1, values2), (double)2.6 );
    }

    TEST(NetMath, crossentropy) {
        std::vector<double> values1 = {1, 0, 0.3};
        std::vector<double> values2 = {0, 1, 0.8};
        EXPECT_EQ( NetMath::crossentropy(values1, values2), (double)70.16654147569186 );
    }

    TEST(NetMath, vanillaupdatefn) {
        Network::deleteNetwork();
        Network::newNetwork();
        Network::getInstance(0)->learningRate = 0.5;
        EXPECT_EQ( NetMath::vanillaupdatefn(0, 10, 10), 15 );
        EXPECT_EQ( NetMath::vanillaupdatefn(0, 10, 20), 20 );
        EXPECT_EQ( NetMath::vanillaupdatefn(0, 10, -30), -5 );
    }

    class GainFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->learningRate = 1;
            testN = new Neuron();
            testN->init(0);
            testN->bias = 0.1;
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete testN;
        }

        Network* net;
        Neuron* testN;
    };

    // Doubles a value when the gain is 2 and learningRate 1
    TEST_F(GainFixture, gain_1) {
        testN->biasGain = 2;
        EXPECT_EQ( NetMath::gain(0, (double)10, (double)5, testN, -1), 20 );
    }

    // Halves a value when the gain is -5 and learningRate 0.1
    TEST_F(GainFixture, gain_2) {
        net->learningRate = 0.1;
        testN->biasGain = -5;
        EXPECT_NEAR( NetMath::gain(0, (double)5, (double)5, testN, -1), 2.5, 1e-6 );
    }

    // Increments a neuron's biasGain by 0.05 when the bias value doesn't change sign
    TEST_F(GainFixture, gain_3) {
        testN->biasGain = 1;
        NetMath::gain(0, (double)0.1, (double)1, testN, -1);
        EXPECT_EQ( testN->biasGain, 1.05 );
    }

    // Does not increase the gain to more than 5
    TEST_F(GainFixture, gain_4) {
        testN->biasGain = 4.99;
        NetMath::gain(0, (double)0.1, (double)1, testN, -1);
        EXPECT_EQ( testN->biasGain, 5 );
    }

    // Multiplies a neuron's bias gain by 0.95 when the value changes sign
    TEST_F(GainFixture, gain_5) {
        net->learningRate = -10;
        testN->biasGain = 1;
        NetMath::gain(0, (double)0.1, (double)1, testN, -1);
        EXPECT_EQ( testN->biasGain, 0.95 );
    }

    // Does not reduce the bias gain to less than 0.5
    TEST_F(GainFixture, gain_6) {
        net->learningRate = -10;
        testN->biasGain = 0.51;
        NetMath::gain(0, (double)0.1, (double)1, testN, -1);
        EXPECT_EQ( testN->biasGain, 0.5 );
    }

    // Increases weight gain the same way as the bias gain
    TEST_F(GainFixture, gain_7) {
        testN->weights = {0.1, 0.1};
        testN->weightGain = {1, 4.99};
        NetMath::gain(0, (double)0.1, (double)1, testN, 0);
        NetMath::gain(0, (double)0.1, (double)1, testN, 1);
        EXPECT_EQ( testN->weightGain[0], 1.05 );
        EXPECT_EQ( testN->weightGain[1], 5 );
    }

    // Decreases weight gain the same way as the bias gain
    TEST_F(GainFixture, gain_8) {
        net->learningRate = -10;
        testN->weights = {0.1, 0.1};
        testN->weightGain = {1, 0.51};
        NetMath::gain(0, (double)0.1, (double)1, testN, 0);
        NetMath::gain(0, (double)0.1, (double)1, testN, 1);
        EXPECT_EQ( testN->weightGain[0], 0.95 );
        EXPECT_EQ( testN->weightGain[1], 0.5 );
    }

    class AdagradFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->learningRate = 2;
            testN = new Neuron();
            testN->init(0);
            testN->biasCache = 0;
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete testN;
        }

        Network* net;
        Neuron* testN;
    };

    // Increments the neuron's biasCache by the square of its deltaBias
    TEST_F(AdagradFixture, adagrad_1) {
        NetMath::adagrad(0, (double)1, (double)3, testN, -1);
        EXPECT_EQ( testN->biasCache, 9 );
    }

    // Returns a new value matching the formula for adagrad
    TEST_F(AdagradFixture, adagrad_2) {
        net->learningRate = 0.5;
        EXPECT_NEAR( NetMath::adagrad(0, (double)1, (double)3, testN, -1), 1.5, 1e-3 );
    }

    // Increments the neuron's weightsCache with the same way as the biasCache
    TEST_F(AdagradFixture, adagrad_3) {
        testN->weightsCache = {0, 1, 2};
        double result1 = NetMath::adagrad(0, (double)1, (double)3, testN, 0);
        double result2 = NetMath::adagrad(0, (double)1, (double)4, testN, 1);
        double result3 = NetMath::adagrad(0, (double)1, (double)2, testN, 2);
        EXPECT_EQ( testN->weightsCache[0], 9 );
        EXPECT_EQ( testN->weightsCache[1], 17 );
        EXPECT_EQ( testN->weightsCache[2], 6 );
        EXPECT_NEAR( result1, 3.0, 1e-2 );
        EXPECT_NEAR( result2, 2.9, 1e-1 );
        EXPECT_NEAR( result3, 2.6, 1e-1 );
    }

    class RMSPropFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->learningRate = 0.5;
            net->rmsDecay = 0.99;
            testN = new Neuron();
            testN->init(0);
            testN->biasCache = 10;
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete testN;
        }

        Network* net;
        Neuron* testN;
    };

    // Sets the cache value to the correct value, following the rmsprop formula
    TEST_F(RMSPropFixture, rmsprop_1) {
        net->learningRate = 2;
        NetMath::rmsprop(0, (double)1, (double)3, testN, -1);
        EXPECT_NEAR(testN->biasCache, 9.99, 1e-3);
    }

    // Returns a new value matching the formula for rmsprop, using this new cache value
    TEST_F(RMSPropFixture, rmsprop_2) {
        EXPECT_NEAR( NetMath::rmsprop(0, (double)1, (double)3, testN, -1), 1.47, 1e-2);
    }

    // Updates the weightsCache the same way as the biasCache
    TEST_F(RMSPropFixture, rmsprop_3) {
        testN->weightsCache = {0, 1, 2};
        double result1 = NetMath::rmsprop(0, (double)1, (double)3, testN, 0);
        double result2 = NetMath::rmsprop(0, (double)1, (double)4, testN, 1);
        double result3 = NetMath::rmsprop(0, (double)1, (double)2, testN, 2);
        EXPECT_NEAR( testN->weightsCache[0], 0.09, 1e-2 );
        EXPECT_NEAR( testN->weightsCache[1], 1.15, 1e-2 );
        EXPECT_NEAR( testN->weightsCache[2], 2.02, 1e-2 );
        EXPECT_NEAR( result1, 6.0, 1e-2 );
        EXPECT_NEAR( result2, 2.9, 0.1 );
        EXPECT_NEAR( result3, 1.7, 0.1 );
    }

    class AdamFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->learningRate = 0.01;
            testN = new Neuron();
            testN->init(0);
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete testN;
        }

        Network* net;
        Neuron* testN;
    };

    // It sets the neuron.m to the correct value, following the formula
    TEST_F(AdamFixture, adam_1) {
        testN->m = 0.1;
        NetMath::adam(0, (double)1, (double)0.2, testN, -1);
        EXPECT_DOUBLE_EQ( testN->m, 0.11 );
    }

    // It sets the neuron.v to the correct value, following the formula
    TEST_F(AdamFixture, adam_2) {
        testN->v = 0.1;
        NetMath::adam(0, (double)1, (double)0.2, testN, -1);
        EXPECT_NEAR( testN->v, 0.09994, 1e-3 );
    }

    // Calculates a value correctly, following the formula
    TEST_F(AdamFixture, adam_3) {
        net->iterations = 2;
        testN->m = 0.121;
        testN->v = 0.045;
        EXPECT_NEAR( NetMath::adam(0, (double)-0.3, (double)0.02, testN, -1), -0.298943, 1e-5 );
    }

    class AdadeltaFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->weightsConfig["limit"] = 0.1;
            net->weightInitFn = &NetMath::uniform;
            net->rho = 0.95;
            testN = new Neuron();
            testN->init(0);
            testN->biasCache = 0.5;
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete testN;
        }

        Network* net;
        Neuron* testN;
    };

    // Sets the neuron.biasCache to the correct value, following the adadelta formula
    TEST_F(AdadeltaFixture, adadelta_1) {
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, -1);
        EXPECT_NEAR( testN->biasCache, 0.477, 1e-3 );
    }

    // Sets the weightsCache to the correct value, following the adadelta formula, same as biasCache
    TEST_F(AdadeltaFixture, adadelta_2) {
        testN->weightsCache = {0.5, 0.75};
        testN->adadeltaCache = {0, 0};
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 0);
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 1);
        EXPECT_NEAR( testN->weightsCache[0], 0.477, 1e-3 );
        EXPECT_NEAR( testN->weightsCache[1], 0.7145, 1e-4 );
    }

    // Creates a value for the bias correctly, following the formula
    TEST_F(AdadeltaFixture, adadelta_3) {
        testN->adadeltaBiasCache = 0.25;
        EXPECT_NEAR( NetMath::adadelta(0, (double)0.5, (double)0.2, testN, -1), 0.64479, 1e-5 );
    }

    // Creates a value for the weight correctly, the same was as the bias
    TEST_F(AdadeltaFixture, adadelta_4) {
        testN->weightsCache = {0.5, 0.75};
        testN->adadeltaCache = {0.1, 0.2};
        EXPECT_NEAR( NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 0), 0.59157, 1e-3 );
        EXPECT_NEAR( NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 1), 0.60581, 1e-3 );
    }

    // Updates the neuron.adadeltaBiasCache with the correct value, following the formula
    TEST_F(AdadeltaFixture, adadelta_5) {
        testN->adadeltaBiasCache = 0.25;
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, -1);
        EXPECT_NEAR( testN->adadeltaBiasCache, 0.2395, 1e-2 );
    }

    // Updates the neuron.adadeltaCache with the correct value, following the formula, same as adadeltaBiasCache
    TEST_F(AdadeltaFixture, adadelta_6) {
        testN->weightsCache = {0.5, 0.75};
        testN->adadeltaCache = {0.1, 0.2};
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 0);
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 1);
        EXPECT_NEAR( testN->adadeltaCache[0], 0.097, 0.1 );
        EXPECT_NEAR( testN->adadeltaCache[1], 0.192, 0.1 );
    }

    TEST(NetMath, sech) {
        EXPECT_DOUBLE_EQ( NetMath::sech(-0.5), 0.886818883970074 );
        EXPECT_DOUBLE_EQ( NetMath::sech(1),    0.6480542736638853 );
    }

    class MaxNormFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->maxNorm = 1;
            net->maxNormTotal = 2.8284271247461903;

            l1 = new FCLayer(0, 1);
            l2 = new FCLayer(0, 2);
            l2->assignPrev(l1);
            net->layers.push_back(l1);
            net->layers.push_back(l2);

            Neuron* n = new Neuron();
            n->weights = {2, 2};
            l2->neurons.push_back(n);
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete l1;
            delete l2;
        }

        Network* net;
        FCLayer* l1;
        FCLayer* l2;
    };

    // Sets the maxNormTotal to 0
    TEST_F(MaxNormFixture, maxNorm_1) {
        NetMath::maxNorm(0);
        EXPECT_EQ(Network::getInstance(0)->maxNormTotal, 0);
    }

    // Scales weights if their L2 exceeds the configured max norm threshold
    TEST_F(MaxNormFixture, maxNorm_2) {
        NetMath::maxNorm(0);
        EXPECT_EQ( l2->neurons[0]->weights[0], 0.7071067811865475 );
        EXPECT_EQ( l2->neurons[0]->weights[1], 0.7071067811865475 );
    }

    // Does not scale weights if their L2 doesn't exceed the configured max norm threshold
    TEST_F(MaxNormFixture, maxNorm_3) {
        net->maxNorm = 1000;
        NetMath::maxNorm(0);
        EXPECT_EQ( l2->neurons[0]->weights[0], 2 );
        EXPECT_EQ( l2->neurons[0]->weights[1], 2 );
    }

    // Returns the same number of values as the size value given
    TEST(NetMath, uniform_1) {
        EXPECT_EQ( NetMath::uniform(0, 0, 10).size(), 10 );
    }

    // Weights are all between -0.1 and +0.1 when the limit is 0.1
    TEST(NetMath, uniform_2) {
        Network::deleteNetwork();
        Network::newNetwork();
        Network::getInstance(0)->weightsConfig["limit"] = 0.1;

        bool ok = true;
        std::vector<double> values = NetMath::uniform(0, 0, 100);

        for (int i=0; i<100; i++) {
            if (values[i] > 0.1 || values[i]<=-0.1) {
                ok = false;
            }
        }

        EXPECT_TRUE( ok );
    }

    // There are some weights bigger than |0.1| when the limit is 1000
    TEST(NetMath, uniform_3) {
        Network::deleteNetwork();
        Network::newNetwork();
        Network::getInstance(0)->weightsConfig["limit"] = 1000;
        bool ok = false;
        std::vector<double> values = NetMath::uniform(0, 0, 1000);

        for (int i=0; i<1000; i++) {
            if (values[i] > 0.1 || values[i]<=-0.1) {
                ok = true;
            }
        }

        EXPECT_TRUE( ok );
    }

    class GaussianFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->weightsConfig["stdDeviation"] = 1;
        }

        virtual void TearDown() {
            Network::deleteNetwork();
        }

        Network* net;
    };

    // Returns the same number of values as the size value given
    TEST_F(GaussianFixture, gaussian_1) {
        EXPECT_EQ( NetMath::gaussian(0, 0, 10).size(), 10 );
    }

    // The standard deviation of the weights is roughly 1 when set to 1
    TEST_F(GaussianFixture, gaussian_2) {
        double std = standardDeviation(NetMath::gaussian(0, 0, 1000));
        EXPECT_LE( std, 1.15 );
        EXPECT_GE( std, 0.85 );
    }

    // The standard deviation of the weights is roughly 5 when set to 5
    TEST_F(GaussianFixture, gaussian_3) {
        net->weightsConfig["stdDeviation"] = 5;
        double std = standardDeviation(NetMath::gaussian(0, 0, 1000));
        EXPECT_LE( std, 1.15 * 5 );
        EXPECT_GE( std, 0.85 * 5 );
    }

    // The mean of the weights is roughly 0 when set to 0
    TEST_F(GaussianFixture, gaussian_4) {
        net->weightsConfig["mean"] = 10;
        std::vector<double> values = NetMath::gaussian(0, 0, 1000);

        double total = 0;

        for (int v=0; v<1000; v++) {
            total += values[v];
        }

        total /= 1000;

        EXPECT_LE( total, 10.1 );
        EXPECT_GE( total, 9.85 );
    }

    // The mean of the weights is roughly 10 when set to 10
    TEST_F(GaussianFixture, gaussian_5) {
        net->weightsConfig["mean"] = 10;
        std::vector<double> values = NetMath::gaussian(0, 0, 1000);

        double total = 0;

        for (int v=0; v<1000; v++) {
            total += values[v];
        }

        total /= 1000;

        EXPECT_LE( total, 10.1 );
        EXPECT_GE( total, 9.85 );
    }


    class LeCunUniformFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            l1 = new FCLayer(0, 1);
            l1->fanIn = 12;
            net->layers.push_back(l1);
        }

        virtual void TearDown() {
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* l1;
    };

    // Returns the same number of values as the size value given
    TEST_F(LeCunUniformFixture, lecununiform_1) {
        EXPECT_EQ( NetMath::lecununiform(0, 0, 10).size(), 10 );
    }

    // Weights are all between -0.5 and +0.5 when fanIn is 12
    TEST_F(LeCunUniformFixture, lecununiform_2) {
        std::vector<double> values = NetMath::lecununiform(0, 0, 1000);

        bool ok = true;

        for (int i=0; i<1000; i++) {
            if (values[i] > 0.5 || values[i] < -0.5 ) {
                ok = false;
            }
        }

        EXPECT_TRUE( ok );
    }

    // Some weights are at values bigger than |0.5| when fanIn is smaller (8)
    TEST_F(LeCunUniformFixture, lecununiform_3) {
        l1->fanIn = 8;
        std::vector<double> values = NetMath::lecununiform(0, 0, 1000);

        bool ok = false;

        for (int i=0; i<1000; i++) {
            if (values[i] > 0.5 || values[i] < -0.5 ) {
                ok = true;
            }
        }

        EXPECT_TRUE( ok );
    }

    class LeCunNormalFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            l1 = new FCLayer(0, 1);
            l1->fanIn = 5;
            net->layers.push_back(l1);
        }

        virtual void TearDown() {
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* l1;
    };

    // Returns the same number of values as the size value given
    TEST_F(LeCunNormalFixture, lecunNormal_1) {
        EXPECT_EQ( NetMath::lecunnormal(0, 0, 10).size(), 10 );
    }

    // The standard deviation of the weights is roughly 0.05 when fanIn is 5
    TEST_F(LeCunNormalFixture, lecunNormal_2) {
        double std = standardDeviation(NetMath::lecunnormal(0, 0, 1000));
        EXPECT_GE( std, 0.4 );
        EXPECT_LE( std, 0.6 );
    }

    // The mean of the weights is roughly 0 when fanIn is 5
    TEST_F(LeCunNormalFixture, lecunNormal_3) {
        std::vector<double> values = NetMath::lecunnormal(0, 0, 1000);

        double total = 0;

        for (int v=0; v<1000; v++) {
            total += values[v];
        }

        total /= 1000;

        EXPECT_GE( total, -0.1 );
        EXPECT_LE( total, 0.1 );
    }

    class XavierUniformFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            l1 = new FCLayer(0, 1);
            net->layers.push_back(l1);
        }

        virtual void TearDown() {
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* l1;
    };

    // Returns the same number of values as the size value given
    TEST_F(XavierUniformFixture, xavierUniform_1) {
        EXPECT_EQ( NetMath::xavieruniform(0, 0, 10).size(), 10 );
    }

    // Weights are all between -0.5 and +0.5 when fanIn is 10 and fanOut is 15
    TEST_F(XavierUniformFixture, xavierUniform_2) {
        l1->fanIn = 10;
        l1->fanOut = 15;
        std::vector<double> values = NetMath::xavieruniform(0, 0, 1000);

        bool ok = true;

        for (int i=0; i<1000; i++) {
            if (values[i] > 0.5 || values[i] < -0.5 ) {
                ok = false;
            }
        }

        EXPECT_TRUE( ok );
    }

    // Inits some weights at values bigger |0.5| when fanIn+fanOut is smaller
    TEST_F(XavierUniformFixture, xavierUniform_3) {
        l1->fanIn = 5;
        l1->fanOut = 5;
        net->layers.push_back(l1);
        std::vector<double> values = NetMath::xavieruniform(0, 0, 1000);

        bool ok = false;

        for (int i=0; i<1000; i++) {
            if (values[i] > 0.05 || values[i] < -0.05 ) {
                ok = true;
            }
        }

        EXPECT_TRUE( ok );
    }

    class XavierNormalFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            l1 = new FCLayer(0, 1);
            l1->fanIn = 5;
            l1->fanOut = 25;
            net->layers.push_back(l1);
        }

        virtual void TearDown() {
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* l1;
    };

    // Returns the same number of values as the size value given
    TEST_F(XavierNormalFixture, xavierNormal_1) {
        EXPECT_EQ( NetMath::xaviernormal(0, 0, 10).size(), 10 );
    }

    // The standard deviation of the weights is roughly 0.25 when fanIn is 5 and fanOut 25
    TEST_F(XavierNormalFixture, xavierNormal_2) {
        double std = standardDeviation(NetMath::xaviernormal(0, 0, 1000));
        EXPECT_GE( std, 0.2 );
        EXPECT_LE( std, 0.3 );
    }

    // The mean of the weights is roughly 0 when fanIn is 5
    TEST_F(XavierNormalFixture, xavierNormal_3) {
        std::vector<double> values = NetMath::xaviernormal(0, 0, 1000);

        double total = 0;

        for (int v=0; v<1000; v++) {
            total += values[v];
        }

        total /= 1000;

        EXPECT_GE( total, -0.1 );
        EXPECT_LE( total, 0.1 );
    }
}

namespace NetUtil_cpp {

    class ShuffleFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            std::tuple<std::vector<double>, std::vector<double> > a;
            std::tuple<std::vector<double>, std::vector<double> > b;
            std::tuple<std::vector<double>, std::vector<double> > c;

            std::get<0>(a) = {1,2,3};
            std::get<1>(a) = {4,5,6};
            values.push_back(a);

            std::get<0>(b) = {7,8,9};
            std::get<1>(b) = {10,11,12};
            values.push_back(b);

            std::get<0>(c) = {13,14,15};
            std::get<1>(c) = {16,17,18};
            values.push_back(c);

            original = values;
        }

        std::vector<std::tuple<std::vector<double>, std::vector<double> > > values;
        std::vector<std::tuple<std::vector<double>, std::vector<double> > > original;
    };

    // Keeps the same number of elements
    TEST_F(ShuffleFixture, shuffle_1) {
        NetUtil::shuffle(values);
        EXPECT_EQ( values.size(), 3 );
    }

    // Changes the order of the elements
    TEST_F(ShuffleFixture, shuffle_2) {
        NetUtil::shuffle(values);
        EXPECT_NE(values, original);
    }

    // Still contains all original values
    TEST_F(ShuffleFixture, shuffle_3) {

        NetUtil::shuffle(values);

        for (int i=0; i<3; i++) {
            bool ok = false;

            for (int j=0; j<3; j++) {

                if (values[i] == original[j]) {
                    ok = true;
                }
            }

            EXPECT_TRUE(ok);
        }
    }

    // Contains no new values
    TEST_F(ShuffleFixture, shuffle_4) {

        NetUtil::shuffle(values);

        for (int i=0; i<3; i++) {

            bool ok = true;

            if (values[i] != original[0] && values[i] != original[1] && values[i] != original[2]) {
                ok = false;
            }

            EXPECT_TRUE(ok);
        }
    }


    class AddZPaddingFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            std::vector<double> testData_0 = {3,5,2,6,8};
            std::vector<double> testData_1 = {9,6,4,3,2};
            std::vector<double> testData_2 = {2,9,3,4,2};
            std::vector<double> testData_3 = {5,8,1,3,7};
            std::vector<double> testData_4 = {4,8,6,4,3};
            testData.push_back(testData_0);
            testData.push_back(testData_1);
            testData.push_back(testData_2);
            testData.push_back(testData_3);
            testData.push_back(testData_4);
        }

        std::vector<std::vector<double> > testData;
    };

    // Returns the same data when zero padding of 0 is given
    TEST_F(AddZPaddingFixture, addZeroPadding_1) {
        std::vector<std::vector<double> > res = NetUtil::addZeroPadding(testData, 0);
        EXPECT_EQ( res, testData );
    }

    // Returns a map with 1 level of 0 values padded, when zero padding of 1 is given
    TEST_F(AddZPaddingFixture, addZeroPadding_2) {
        std::vector<std::vector<double> > res = NetUtil::addZeroPadding(testData, 1);
        std::vector<double> zeroRow = {0,0,0,0,0,0,0};

        EXPECT_EQ( res.size(), 7 );
        EXPECT_EQ( res[0], zeroRow );
        EXPECT_NE( res[1], zeroRow );
        EXPECT_NE( res[res.size()-2], zeroRow );
        EXPECT_EQ( res[res.size()-1], zeroRow );
    }

    // Returns a map with 3 levels of 0 values padded, when zero padding of 3 is given
    TEST_F(AddZPaddingFixture, addZeroPadding_3) {
        std::vector<std::vector<double> > res = NetUtil::addZeroPadding(testData, 3);
        std::vector<double> zeroRow = {0,0,0,0,0,0,0,0,0,0,0};

        EXPECT_EQ( res.size(), 11 );
        EXPECT_EQ( res[0], zeroRow );
        EXPECT_EQ( res[1], zeroRow );
        EXPECT_EQ( res[2], zeroRow );
        EXPECT_NE( res[3], zeroRow );
        EXPECT_NE( res[res.size()-4], zeroRow );
        EXPECT_EQ( res[res.size()-3], zeroRow );
        EXPECT_EQ( res[res.size()-2], zeroRow );
        EXPECT_EQ( res[res.size()-1], zeroRow );
    }

    // Keeps the same data, apart for the zeroes
    TEST_F(AddZPaddingFixture, addZeroPadding_4) {
        std::vector<std::vector<double> > res = NetUtil::addZeroPadding(testData, 1);

        res[1].erase(res[1].begin());
        res[1].pop_back();
        res[2].erase(res[2].begin());
        res[2].pop_back();
        res[3].erase(res[3].begin());
        res[3].pop_back();
        res[4].erase(res[4].begin());
        res[4].pop_back();
        res[5].erase(res[5].begin());
        res[5].pop_back();

        EXPECT_EQ( res[1], testData[0] );
        EXPECT_EQ( res[2], testData[1] );
        EXPECT_EQ( res[3], testData[2] );
        EXPECT_EQ( res[4], testData[3] );
        EXPECT_EQ( res[5], testData[4] );
    }
}


int main (int argc, char** argv) {
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}