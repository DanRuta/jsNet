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

    // Assigns prevLayers to layers accordingly
    TEST(Network, joinLayers_1) {
        Network::deleteNetwork();
        Network::newNetwork();
        Network::getInstance(0)->weightsConfig["limit"] = 0.1;
        Network::getInstance(0)->weightInitFn = &NetMath::uniform;
        Network::getInstance(0)->layers.push_back(new FCLayer(0, 3));
        Network::getInstance(0)->layers.push_back(new FCLayer(0, 3));
        Network::getInstance(0)->layers.push_back(new FCLayer(0, 3));
        Network::getInstance(0)->joinLayers();

        EXPECT_EQ( Network::getInstance(0)->layers[1]->prevLayer, Network::getInstance(0)->layers[0] );
        EXPECT_EQ( Network::getInstance(0)->layers[2]->prevLayer, Network::getInstance(0)->layers[1] );
    }

    // Assigns nextLayers to layers accordingly
    TEST(Network, joinLayers_2) {
        EXPECT_EQ( Network::getInstance(0)->layers[0]->nextLayer, Network::getInstance(0)->layers[1] );
        EXPECT_EQ( Network::getInstance(0)->layers[1]->nextLayer, Network::getInstance(0)->layers[2] );
    }

    // Assigns fanIn and fanOut correctly
    TEST(Network, joinLayers_3) {
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

        net->layers[0]->neurons = {new Neuron(), new Neuron(), new Neuron()};
        net->forward(testInput);

        delete l3;
    }

    // Sets the first layer's neurons' activations to the input given
    TEST_F(ForwardFixture, forward_2) {

        net->layers[0]->neurons = {new Neuron(), new Neuron(), new Neuron()};
        net->forward(testInput);

        std::vector<double> expected = {1, 2, 3};
        EXPECT_EQ( l1->actvns, expected );
    }


    // Returns a vector of softmax-ed sums in the last layer
    TEST_F(ForwardFixture, forward_3) {

        l2->sums = {1,2};

        net->layers[0]->neurons = {new Neuron(), new Neuron(), new Neuron()};
        std::vector<double> returned = net->forward(testInput);
        std::vector<double> actualValues = NetMath::softmax({1, 2});

        EXPECT_EQ( returned, actualValues );
    }

    // Calls the last layer's backward function with errors, and every other layer'ss except the first with an empty vector
    TEST(Network, backward) {
        Network::deleteNetwork();
        Network::newNetwork();

        MockLayer* l1 = new MockLayer(0, 3);
        MockLayer* l2 = new MockLayer(0, 3);
        MockLayer* l3 = new MockLayer(0, 3);

        Network::getInstance(0)->layers.push_back(l1);
        Network::getInstance(0)->layers.push_back(l2);
        Network::getInstance(0)->layers.push_back(l3);

        std::vector<double> errors = {1,2,3};

        EXPECT_CALL(*l1, backward(false)).Times(0);
        EXPECT_CALL(*l2, backward(false)).Times(1);
        EXPECT_CALL(*l3, backward(true)).Times(1);

        Network::getInstance(0)->backward();

        delete l1;
        delete l2;
        delete l3;
        Network::deleteNetwork();
    }

    // Forward, backward, resetDeltaWeights and applyDeltaWeights functions are called appropriately
    TEST(Network, train) {
        Network::deleteNetwork();
        Network::newNetwork();
        Network::getInstance(0)->costFunction = NetMath::meansquarederror;

        std::vector<std::tuple<std::vector<double>, std::vector<double> > > trainingData = {};
        std::tuple<std::vector<double>, std::vector<double> > data;
        std::get<0>(data) = {};
        std::get<1>(data) = {};
        trainingData.push_back(data);
        trainingData.push_back(data);

        Network::getInstance(0)->trainingData = trainingData;
        Network::getInstance(0)->miniBatchSize = 1;

        MockLayer* l1 = new MockLayer(0, 3);
        MockLayer* l2 = new MockLayer(0, 3);
        MockLayer* l3 = new MockLayer(0, 3);

        Network::getInstance(0)->layers.push_back(l1);
        Network::getInstance(0)->layers.push_back(l2);
        Network::getInstance(0)->layers.push_back(l3);

        l1->neurons = {};
        l3->neurons = {};

        EXPECT_CALL(*l1, forward()).Times(0);
        EXPECT_CALL(*l2, forward()).Times(1);
        EXPECT_CALL(*l3, forward()).Times(1);
        EXPECT_CALL(*l1, backward(false)).Times(0);
        EXPECT_CALL(*l2, backward(false)).Times(1);
        EXPECT_CALL(*l3, backward(true)).Times(1);
        EXPECT_CALL(*l1, resetDeltaWeights()).Times(0);
        EXPECT_CALL(*l2, resetDeltaWeights()).Times(1);
        EXPECT_CALL(*l3, resetDeltaWeights()).Times(1);
        EXPECT_CALL(*l1, applyDeltaWeights()).Times(0);
        EXPECT_CALL(*l2, applyDeltaWeights()).Times(1);
        EXPECT_CALL(*l3, applyDeltaWeights()).Times(1);

        Network::getInstance(0)->train(1, 0);

        delete l1;
        delete l2;
        delete l3;
        Network::deleteNetwork();
    }

    // The forward and backward functions are called appropriately
    TEST(Network, test) {
        Network::deleteNetwork();
        Network::newNetwork();
        Network::getInstance(0)->costFunction = NetMath::meansquarederror;

        std::vector<std::tuple<std::vector<double>, std::vector<double> > > testData = {};
        std::tuple<std::vector<double>, std::vector<double> > data;
        std::get<0>(data) = {};
        std::get<1>(data) = {};
        testData.push_back(data);
        testData.push_back(data);

        Network::getInstance(0)->testData = testData;

        MockLayer* l1 = new MockLayer(0, 3);
        MockLayer* l2 = new MockLayer(0, 3);
        MockLayer* l3 = new MockLayer(0, 3);

        Network::getInstance(0)->layers.push_back(l1);
        Network::getInstance(0)->layers.push_back(l2);
        Network::getInstance(0)->layers.push_back(l3);

        l1->neurons = {};
        l3->neurons = {};

        EXPECT_CALL(*l1, forward()).Times(0);
        EXPECT_CALL(*l2, forward()).Times(1);
        EXPECT_CALL(*l3, forward()).Times(1);
        EXPECT_CALL(*l1, backward(false)).Times(0);
        EXPECT_CALL(*l2, backward(false)).Times(0);
        EXPECT_CALL(*l3, backward(false)).Times(0);

        Network::getInstance(0)->test(1, 0);

        delete l1;
        delete l2;
        delete l3;
        Network::deleteNetwork();
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

    // Assigns the type as FC
    TEST(FCLayer, constructor) {
        FCLayer* layer = new FCLayer(0, 1);
        EXPECT_EQ(layer->type, "FC");
    }

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

    // Sets the bias of every neuron to 0
    TEST_F(InitFixture, init_2) {
        l2->init(1);

        for (int n=0; n<5; n++) {
            EXPECT_EQ( l2->biases[n], 1 );
        }
    }

    // Inits the neurons' weights vector with as many weights as there are neurons in the prev layer. (none in first layer)
    TEST_F(InitFixture, init_3) {
        l1->init(0);
        l2->init(1);

        EXPECT_EQ( l1->weights.size(), 0 );

        for (int n=0; n<5; n++) {
            EXPECT_EQ( l2->weights[n].size(), 2 );
        }
    }

    // Inits the neurons' weights vector with as many weights as there are outgoing values in every filter in a prev Conv layer
    TEST_F(InitFixture, init_4) {

        ConvLayer* c = new ConvLayer(0, 2);
        c->outMapSize = 3;
        c->filters = {};
        c->filters.push_back(new Filter());
        c->filters.push_back(new Filter());
        c->size = 2;

        l2->prevLayer = c;
        l2->init(1);

        for (int n=0; n<5; n++) {
            EXPECT_EQ( l2->weights[n].size(), 18 );
        }

        delete c;
    }

    // Inits the neurons' weights vector with as many weights as there are outgoing values in a prev Pool layer
    TEST_F(InitFixture, init_5) {

        PoolLayer* p = new PoolLayer(0, 2);
        p->outMapSize = 3;
        std::vector<std::vector<double> > testData = {{1}};
        p->activations = {testData, testData, testData, testData, testData};

        l2->prevLayer = p;
        l2->init(1);

        for (int n=0; n<5; n++) {
            EXPECT_EQ( l2->weights[n].size(), 45 );
        }

        delete p;
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
            l2->hasActivation = true;
            l2->prevLayer = l1;
            l2->netInstance = 1;
            l1->init(0);
            l2->init(1);
            l2->activation = &NetMath::sigmoid<Neuron>;
            l1->actvns = {1,2};
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
            l2->weights[n] = {1,2};
        }
        l2->biases = {0,1,2,1,1};

        net->isTraining= false;
        net->dropout = 1;

        l2->forward();

        std::vector<double> expected = {5,6,7,1,1};

        EXPECT_EQ( l2->sums, expected );

        // Check that it SETS it, and doesn't increment it
        l2->forward();

        EXPECT_FALSE( net->isTraining );
        EXPECT_EQ( l2->sums, expected );
    }

    // Sets the layer's neurons' activation to the result of the activation function
    TEST_F(FCForwardFixture, forward_2) {

        for (int n=0; n<3; n++) {
            l2->weights[n] = {1,2};
        }
        l2->biases = {0,1,2};

        net->isTraining = false;
        net->dropout = 1;
        l2->forward();

        EXPECT_FALSE( net->isTraining );

        EXPECT_DOUBLE_EQ( l2->actvns[0], 0.9933071490757153 );
        EXPECT_DOUBLE_EQ( l2->actvns[1], 0.9975273768433653 );
        EXPECT_DOUBLE_EQ( l2->actvns[2], 0.9990889488055994 );
    }

    // Sets the layer's neurons' activation to the sum when there is no activation function
    TEST_F(FCForwardFixture, forward_3) {

        l2->hasActivation = false;

        for (int n=0; n<3; n++) {
            l2->weights[n] = {1,2};
        }
        l2->biases = {0,1,2};

        net->isTraining = false;
        net->dropout = 1;
        l2->forward();

        EXPECT_FALSE( net->isTraining );
        EXPECT_DOUBLE_EQ( l2->actvns[0], 5 );
        EXPECT_DOUBLE_EQ( l2->actvns[1], 6 );
        EXPECT_DOUBLE_EQ( l2->actvns[2], 7 );
    }

    // Sets the neurons' dropped value to true and activation to 0 if the net is training and dropout is set to 0
    TEST_F(FCForwardFixture, forward_4) {

        l2->sums = {0,0,0};
        l2->biases = {0,1,2};

        net->dropout = 0;
        net->isTraining = true;
        l2->forward();

        EXPECT_TRUE( l2->neurons[0]->dropped );
        EXPECT_TRUE( l2->neurons[1]->dropped );
        EXPECT_TRUE( l2->neurons[2]->dropped );

        EXPECT_EQ( l2->actvns[0], 0 );
        EXPECT_EQ( l2->actvns[1], 0 );
        EXPECT_EQ( l2->actvns[2], 0 );

        std::vector<double> expected = {0,0,0};

        EXPECT_EQ( l2->sums, expected );
    }

    // Does not set neurons to dropped if the net is not training
    TEST_F(FCForwardFixture, forward_5) {
        for (int n=0; n<3; n++) {
            l2->weights[n] = {1,2};
        }
        l2->actvns = {0,0,0};
        l2->biases = {0,1,2};

        net->dropout = 1;
        net->isTraining = false;
        l2->forward();

        EXPECT_FALSE( l2->neurons[0]->dropped );
        EXPECT_FALSE( l2->neurons[1]->dropped );
        EXPECT_FALSE( l2->neurons[2]->dropped );

        EXPECT_DOUBLE_EQ( l2->actvns[0], 0.9933071490757153 );
        EXPECT_DOUBLE_EQ( l2->actvns[1], 0.9975273768433653 );
        EXPECT_DOUBLE_EQ( l2->actvns[2], 0.9990889488055994 );
    }

    // Divides the activation values by the dropout
    TEST_F(FCForwardFixture, forward_6) {
        for (int n=0; n<3; n++) {
            l2->weights[n] = {1,2};
        }
        l2->biases = {0,1,2};

        net->dropout = 0.5;
        net->isTraining = false;
        l2->forward();

        EXPECT_FALSE( net->isTraining );
        EXPECT_DOUBLE_EQ( l2->actvns[0], 0.9933071490757153 * 2 );
        EXPECT_DOUBLE_EQ( l2->actvns[1], 0.9975273768433653 * 2 );
        EXPECT_DOUBLE_EQ( l2->actvns[2], 0.9990889488055994 * 2 );
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
            l2->hasActivation = true;
            l2->activation = &NetMath::sigmoid<Neuron>;
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

    // Sets the neurons' derivatives to the activation prime of their sum, when no expected data is passed
    TEST_F(FCBackwardFixture, backward_2) {

        l2->sums = {0,1,0};
        l2->neurons[0]->dropped = false;
        l2->neurons[1]->dropped = false;
        l2->neurons[2]->dropped = false;
        l2->backward(false);

        EXPECT_EQ( l2->neurons[0]->derivative, 0.25 );
        EXPECT_DOUBLE_EQ( l2->neurons[1]->derivative, 0.19661193324148185 );
        EXPECT_EQ( l2->neurons[2]->derivative, 0.25 );
    }

    // Sets the neurons' derivatives to 1 when no activation function is configured
    TEST_F(FCBackwardFixture, backward_3) {

        l2->hasActivation = false;

        l2->sums = {0,0,0};
        l2->neurons[0]->dropped = false;
        l2->neurons[1]->dropped = false;
        l2->neurons[2]->dropped = false;
        l2->backward(false);

        EXPECT_EQ( l2->neurons[0]->derivative, 1 );
        EXPECT_EQ( l2->neurons[1]->derivative, 1 );
        EXPECT_EQ( l2->neurons[2]->derivative, 1 );
    }

    // Sets the neurons' errors to their derivative multiplied by weighted errors in next layer, when no expected data is passed
    TEST_F(FCBackwardFixture, backward_4) {

        l2->sums = {0.5,0.5,0.5};
        l2->neurons[0]->dropped = false;
        l2->neurons[1]->dropped = false;
        l2->neurons[2]->dropped = false;

        for (int i=0; i<4; i++) {
            l3->weights[i] = {1,1,1,1};
        }
        l3->errs = {0.5, 0.5, 0.5, 0.5};

        l2->backward(false);
        std::vector<double> expected = {0.470007424403189, 0.470007424403189, 0.470007424403189};
        EXPECT_EQ( l2->errs, expected );
    }

    // Increments each of its delta weights by its error * the respective weight's neuron's activation
    TEST_F(FCBackwardFixture, backward_5) {

        Network::getInstance(0)->l2 = 0;

        l2->actvns = {0.5, 0.5, 0.5};
        l3->neurons[0]->dropped = false;
        l3->neurons[1]->dropped = false;
        l3->neurons[2]->dropped = false;
        l3->neurons[3]->dropped = false;

        l3->errs = {0.5,1.5,2.5,3.5};
        l3->backward(true);

        for (int n=0; n<4; n++) {
            EXPECT_EQ( l3->deltaWeights[n][0], 0.25 + n * 0.5 );
            EXPECT_EQ( l3->deltaWeights[n][1], 0.25 + n * 0.5 );
            EXPECT_EQ( l3->deltaWeights[n][2], 0.25 + n * 0.5 );
        }
    }

    // Increments the neurons' deltaBias to their errors
    TEST_F(FCBackwardFixture, backward_6) {
        std::vector<double> expected = {1,1,3};

        l2->neurons[0]->deltaBias = 1;
        l2->neurons[1]->deltaBias = 1;
        l2->neurons[2]->deltaBias = 1;
        l2->neurons[0]->dropped = false;
        l2->neurons[1]->dropped = false;
        l2->neurons[2]->dropped = false;

        l2->errs = expected;
        l2->backward(true);

        expected = {1,1,3};
        EXPECT_EQ( l2->errs, expected );

        EXPECT_EQ( l2->neurons[0]->deltaBias, 2 );
        EXPECT_EQ( l2->neurons[1]->deltaBias, 2 );
        EXPECT_EQ( l2->neurons[2]->deltaBias, 4 );
    }

    // Sets the neurons' error and deltaBias values to 0 when they are dropped
    TEST_F(FCBackwardFixture, backward_7) {
        std::vector<double> expected = {1,2,3};

        l2->actvns = {0, 1, 0};

        l2->neurons[0]->dropped = true;
        l2->neurons[1]->dropped = true;
        l2->neurons[2]->dropped = true;
        l2->neurons[0]->deltaBias = 456;
        l2->neurons[2]->deltaBias = 456;
        l2->neurons[1]->deltaBias = 456;

        l2->errs = expected;
        l2->backward(true);

        expected = {0,0,0};
        EXPECT_EQ( l2->errs, expected );

        EXPECT_EQ( l2->neurons[0]->deltaBias, 0 );
        EXPECT_EQ( l2->neurons[1]->deltaBias, 0 );
        EXPECT_EQ( l2->neurons[2]->deltaBias, 0 );
    }

    // Increments the deltaWeights by the orig value, multiplied by the l2 amount * existing deltaWeight value
    TEST_F(FCBackwardFixture, backward_8) {
        std::vector<double> expected = {0.05, 0.05, 0.05, 0.05};
        net->l2 = 0.001;

        l2->actvns = {0.5, 0.5, 0.5};
        l3->neurons[0]->dropped = false;
        l3->neurons[1]->dropped = false;
        l3->neurons[2]->dropped = false;
        l3->neurons[3]->dropped = false;

        for (int i=0; i<4; i++) {
            l3->deltaWeights[i][0] = 0.25;
            l3->deltaWeights[i][1] = 0.25;
            l3->deltaWeights[i][2] = 0.25;
        }

        l3->errs = expected;
        l3->backward(true);

        EXPECT_EQ( net->l2, 0.001 );

        for (int n=0; n<4; n++) {
            EXPECT_NEAR( l3->deltaWeights[n][0], 0.27500625, 1e-2 );
            EXPECT_NEAR( l3->deltaWeights[n][1], 0.27500625, 1e-2 );
            EXPECT_NEAR( l3->deltaWeights[n][2], 0.27500625, 1e-2 );
        }
    }

    // Increments the deltaWeights by the orig value, multiplied by the l1 amount * existing deltaWeight value
    TEST_F(FCBackwardFixture, backward_9) {
        std::vector<double> expected = {0.05, 0.05, 0.05, 0.05};
        net->l1 = 0.005;

        l2->actvns = {0.5, 0.5, 0.5};
        l3->neurons[0]->dropped = false;
        l3->neurons[1]->dropped = false;
        l3->neurons[2]->dropped = false;
        l3->neurons[3]->dropped = false;

        for (int i=0; i<4; i++) {
            l3->deltaWeights[i] = {0.25, 0.25, 0.25, 0.25};
        }

        l3->errs = expected;
        l3->backward(true);

        for (int n=0; n<4; n++) {
            EXPECT_NEAR( l3->deltaWeights[n][0], 0.275031, 1e-2 );
            EXPECT_NEAR( l3->deltaWeights[n][1], 0.275031, 1e-2 );
            EXPECT_NEAR( l3->deltaWeights[n][2], 0.275031, 1e-2 );
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
            net = Network::getInstance(0);
            net->l2 = 0.001;
            net->l2Error = 0;
            net->l1 = 0.005;
            net->miniBatchSize = 1;
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

            l2->weights = {{0.25, 0.25}, {0.25, 0.25}, {0.25, 0.25}};
            l2->deltaWeights = {{0.25, 0.25}, {0.25, 0.25}, {0.25, 0.25}};
            l3->weights = {{0.25, 0.25, 0.25}};
            l4->weights = {{0.25}};
            l4->deltaWeights = {{0.25}};
            l4->biases = {1, 1};
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
            l2->neurons[n]->deltaBias = n*2;
        }

        l2->biases = {0, 1, 2};

        l2->applyDeltaWeights();

        EXPECT_EQ( l2->biases[0], 0 );
        EXPECT_EQ( l2->biases[1], 3 );
        EXPECT_EQ( l2->biases[2], 6 );
    }

    // Increments the l2Error by the l2 formula applied to all weights
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_3) {
        Network::getInstance(l2->netInstance)->l2Error = 0;

        for (int n=0; n<3; n++) {
            l2->neurons[n]->deltaBias = n*2;
        }

        l2->biases = {0, 1, 2};
        l2->applyDeltaWeights();
        EXPECT_NEAR( Network::getInstance(l2->netInstance)->l2Error, 0.0001875 , 1e-6 );
    }

    // Increments the l1Error by the l1 formula applied to all weights
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_4) {
        Network::getInstance(l2->netInstance)->l1Error = 0;

        for (int n=0; n<3; n++) {
            l2->neurons[n]->deltaBias = n*2;
        }

        l2->biases = {0, 1, 2};
        l2->applyDeltaWeights();
        EXPECT_NEAR( Network::getInstance(l2->netInstance)->l1Error, 0.0075 , 1e-6 );
    }

    // Increments the bias by the deltaBias following the gain function
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_5) {

        net->updateFnIndex = 1;

        for (int n=0; n<3; n++) {
            l2->neurons[n]->deltaBias = n*2;
            l2->neurons[n]->biasGain = 1;
            l2->neurons[n]->weightGain = {0.25, 0.25};
        }

        l2->biases = {0, 1, 2};

        l2->applyDeltaWeights();
        l2->applyDeltaWeights();

        EXPECT_EQ( l2->biases[0], 0 );
        EXPECT_EQ( l2->biases[1], 5.1 );
        EXPECT_EQ( l2->biases[2], 10.2 );
    }

    // Increments the bias by the deltaBias following the adagrad function
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_6) {

        net->updateFnIndex = 2;

        for (int n=0; n<3; n++) {
            l2->neurons[n]->deltaBias = n*2;
            l2->neurons[n]->biasCache = 1;
            l2->neurons[n]->weightsCache = {0.25, 0.25};
        }

        l2->biases = {0, 1, 2};

        l2->applyDeltaWeights();

        EXPECT_NEAR( (double)l2->biases[0], (double)0, 1e-3 );
        EXPECT_NEAR( (double)l2->biases[1], (double)1.89443, 1e-3 );
        EXPECT_NEAR( (double)l2->biases[2], (double)2.97014, 1e-3 );
    }

    // Increments the bias by the deltaBias following the rmsprop function
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_7) {

        net->updateFnIndex = 3;
        net->rmsDecay = 0.99;

        for (int n=0; n<3; n++) {
            l2->neurons[n]->deltaBias = n*2;
            l2->neurons[n]->biasCache = 1;
            l2->neurons[n]->weightsCache = {0.25, 0.25};
        }

        l2->biases = {0, 1, 2};

        l2->applyDeltaWeights();

        EXPECT_NEAR( (double)l2->biases[0], (double)0, 1e-3 );
        EXPECT_NEAR( (double)l2->biases[1], (double)2.97065, 1e-3 );
        EXPECT_NEAR( (double)l2->biases[2], (double)5.73006, 1e-3 );
    }

    // Increments the bias by the deltaBias following the adam function
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_8) {

        net->updateFnIndex = 4;
        net->l1 = 0;
        net->l2 = 0;

        for (int n=0; n<3; n++) {
            l2->neurons[n]->deltaBias = n*2;
            l2->neurons[n]->m = 0;
            l2->neurons[n]->v = 0;
        }

        l2->biases = {0, 1, 2};

        l2->applyDeltaWeights();

        EXPECT_NEAR( (double)l2->biases[0], (double)1.21006, 1e-3 );
        EXPECT_NEAR( (double)l2->biases[1], (double)2.19524, 1e-3 );
        EXPECT_NEAR( (double)l2->biases[2], (double)3.10259, 1e-3 );
    }

    // Increments the bias by the deltaBias following the adadelta function
    TEST_F(FCApplyDeltaWeightsFixture, applyDeltaWeights_9) {

        net->updateFnIndex = 5;
        net->rho = 0.95;

        for (int n=0; n<3; n++) {
            l2->neurons[n]->deltaBias = n*2;
            l2->neurons[n]->biasCache = 1;
            l2->neurons[n]->adadeltaBiasCache = 1;
            l2->neurons[n]->weightsCache = {0.25, 0.25};
            l2->neurons[n]->adadeltaCache = {0.25, 0.25};

        }

        l2->biases = {0, 1, 2};

        l2->applyDeltaWeights();
        l2->applyDeltaWeights();

        EXPECT_NEAR( (double)l2->biases[0], (double)0, 1e-3 );
        EXPECT_NEAR( (double)l2->biases[1], (double)4.75154, 1e-3 );
        EXPECT_NEAR( (double)l2->biases[2], (double)8.39574, 1e-3 );
    }


    // Clears the neurons' deltaBias
    TEST(FCLayer, resetDeltaWeights_1) {
        Network::deleteNetwork();
        Network::newNetwork();
        FCLayer* l1 = new FCLayer(0, 2);
        FCLayer* l2 = new FCLayer(0, 3);
        l2->assignPrev(l1);
        l2->prevLayer = l1;
        l2->neurons.push_back(new Neuron());
        l2->neurons.push_back(new Neuron());
        l2->neurons.push_back(new Neuron());
        l2->weights = {{}, {}, {}};
        l2->deltaWeights = {{}, {}, {}};

        for (int n=1; n<3; n++) {
            l2->neurons[n]->deltaBias = 1;
        }

        l2->resetDeltaWeights();

        for (int n=1; n<3; n++) {
            EXPECT_EQ( l2->neurons[n]->deltaBias, 0 );
        }

        delete l1;
        delete l2;
        Network::deleteNetwork();
    }

    // Sets all deltaWeight values to 0
    TEST(FCLayer, resetDeltaWeights_2) {
        Network::deleteNetwork();
        Network::newNetwork();
        FCLayer* l1 = new FCLayer(0, 2);
        FCLayer* l2 = new FCLayer(0, 3);
        l2->assignPrev(l1);
        l2->prevLayer = l1;
        l2->neurons.push_back(new Neuron());
        l2->neurons.push_back(new Neuron());
        l2->neurons.push_back(new Neuron());

        l2->weights = {{1,2}, {1,2}, {1,2}};
        l2->deltaWeights = {{1,2}, {1,2}, {1,2}};
        std::vector<std::vector<double> > expected = {{0,0}, {0,0}, {0,0}};

        l2->resetDeltaWeights();

        EXPECT_EQ( l2->deltaWeights, expected );

        delete l1;
        delete l2;
        Network::deleteNetwork();
    }
}

namespace ConvLayer_cpp {

    // Assigns the type as Conv
    TEST(ConvLayer, constructor) {
        ConvLayer* layer = new ConvLayer(0, 1);
        EXPECT_EQ( layer->type, "Conv" );
        delete layer;
    }

    // Assigns the given layer pointer to this layer's nextLayer
    TEST(ConvLayer, assignNext) {
        ConvLayer* l1 = new ConvLayer(0, 1);
        ConvLayer* l2 = new ConvLayer(0, 1);

        l1->assignNext(l2);
        EXPECT_EQ( l1->nextLayer, l2 );

        delete l1;
        delete l2;
    }

    // Assigns the given layer pointer to this layer's prevLayer
    TEST(ConvLayer, assignPrev_1) {
        ConvLayer* l1 = new ConvLayer(0, 1);
        ConvLayer* l2 = new ConvLayer(0, 1);

        l2->assignPrev(l1);
        EXPECT_EQ( l2->prevLayer, l1 );

        delete l1;
        delete l2;
    }

    // Adds as many filters to the layer->filters vector as the size of the layer
    TEST(ConvLayer, assignPrev_2) {
        ConvLayer* l1 = new ConvLayer(0, 1);
        ConvLayer* l2 = new ConvLayer(0, 5);

        EXPECT_EQ( l2->filters.size(), 0 );
        l2->assignPrev(l1);
        EXPECT_EQ( l2->filters.size(), 5 );

        delete l1;
        delete l2;
    }

    class ConvLayerInitFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->dropout = 1;
            net->weightInitFn = &NetMath::uniform;
            layer = new ConvLayer(0, 4);
            prevFC = new FCLayer(0, 10);
            layer->size = 4;
            layer->filterSize = 3;
            layer->zeroPadding = 0;
            layer->stride = 1;
            layer->channels = 2;
            layer->outMapSize = 3;
        }

        virtual void TearDown() {
            delete layer;
            delete prevFC;
            Network::deleteNetwork();
        }

        ConvLayer* layer;
        FCLayer* prevFC;
        Network* net;
    };

    // Assigns a volume of weights to each filter
    TEST_F(ConvLayerInitFixture, init_1) {
        layer->assignPrev(prevFC);
        layer->init(1);

        for (int f=0; f<4; f++) {
            EXPECT_EQ( layer->filters[f]->weights.size(), 2 );
            EXPECT_EQ( layer->filters[f]->weights[0].size(), 3 );
            EXPECT_EQ( layer->filters[f]->weights[0][0].size(), 3 );
        }
    }

    // Creates an activationMap map with 0 values
    TEST_F(ConvLayerInitFixture, init_2) {
        layer->assignPrev(prevFC);
        layer->init(1);

        std::vector<std::vector<double> > expected = { {0,0,0}, {0,0,0}, {0,0,0} };

        for (int f=0; f<4; f++) {
            EXPECT_EQ( layer->filters[f]->activationMap, expected );
        }
    }

    // Creates an errorMap map with 0 values
    TEST_F(ConvLayerInitFixture, init_3) {
        layer->assignPrev(prevFC);
        layer->init(1);

        std::vector<std::vector<double> > expected = { {0,0,0}, {0,0,0}, {0,0,0} };

        for (int f=0; f<4; f++) {
            EXPECT_EQ( layer->filters[f]->activationMap, expected );
        }
    }

    // Sets the bias to 0
    TEST_F(ConvLayerInitFixture, init_4) {
        layer->assignPrev(prevFC);
        layer->init(1);

        for (int f=0; f<4; f++) {
            EXPECT_EQ( layer->filters[f]->bias, 1 );
        }
    }

    // Creates a dropout when net->dropout is not 1
    TEST_F(ConvLayerInitFixture, init_5) {
        net->dropout = 0.5;
        layer->assignPrev(prevFC);
        layer->init(1);
        std::vector<std::vector<double> > expected = { {false,false,false}, {false,false,false}, {false,false,false} };

        for (int f=0; f<4; f++) {
            EXPECT_EQ( layer->filters[f]->activationMap, expected );
        }
    }

    // Does not create a dropout when net->dropout is 1
    TEST_F(ConvLayerInitFixture, init_6) {
        net->dropout = 1;
        layer->assignPrev(prevFC);
        layer->init(1);

        for (int f=0; f<4; f++) {
            EXPECT_EQ( layer->filters[f]->dropoutMap.size(), 0 );
        }
    }

    class ConvForwardFixture : public ::testing::Test {
    public:
        virtual void SetUp () {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->weightInitFn = &NetMath::uniform;
            net->isTraining = false;
            prevLayer = new FCLayer(0, 75);
            prevLayer->size = 75;
            layer = new ConvLayer(0, 1);
            layer->size = 3;
            layer->filterSize = 3;
            layer->zeroPadding = 1;
            layer->stride = 1;
            layer->channels = 3;
            layer->outMapSize = 5;
            layer->hasActivation = false;
            prevLayer->init(0);
            layer->assignPrev(prevLayer);
            layer->init(1);
            expected = {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}};
        }

        virtual void TearDown () {
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* prevLayer;
        ConvLayer* layer;
        std::vector<std::vector<double> > expected;
    };

    // Sets the filter.sumMap of each filter to a map with spacial dimension equal to the output volume's (layer.outMapSize)
    TEST_F(ConvForwardFixture, forward_1) {
        layer->forward();
        EXPECT_EQ( layer->filters[0]->sumMap.size(), 5 );
        EXPECT_EQ( layer->filters[0]->sumMap[0].size(), 5 );
    }

    // Sets the filter.activationMap values to zero if when dropping out
    TEST_F(ConvForwardFixture, forward_2) {
        net->dropout = 0;
        net->isTraining = true;
        layer->forward();

        EXPECT_EQ( layer->filters[0]->activationMap, expected );
        EXPECT_EQ( layer->filters[1]->activationMap, expected );
        EXPECT_EQ( layer->filters[2]->activationMap, expected );
    }

    // Doesn't do any dropout if the layer state is not training
    TEST_F(ConvForwardFixture, forward_3) {
        net->dropout = 0;
        net->isTraining = false;
        layer->forward();

        EXPECT_NE( layer->filters[0]->activationMap, expected );
        EXPECT_NE( layer->filters[1]->activationMap, expected );
        EXPECT_NE( layer->filters[2]->activationMap, expected );
    }

    // Doesn't do any dropout if the dropout is set to 1
    TEST_F(ConvForwardFixture, forward_4) {
        net->dropout = 1;
        net->isTraining = true;
        layer->forward();

        EXPECT_NE( layer->filters[0]->activationMap, expected );
        EXPECT_NE( layer->filters[1]->activationMap, expected );
        EXPECT_NE( layer->filters[2]->activationMap, expected );
    }

    // Gives each filter's activationMap values a value
    TEST_F(ConvForwardFixture, forward_5) {

        for (int f=0; f<layer->filters.size(); f++) {
            layer->filters[f]->activationMap = {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}};
        }

        net->isTraining = false;
        layer->forward();

        for (int f=0; f<layer->filters.size(); f++) {
            EXPECT_NE( layer->filters[f]->activationMap, expected );
        }
    }

    // It does not pass sum map values through an activation function if it is set to not be used
    TEST_F(ConvForwardFixture, forward_6) {

        std::vector<std::vector<double> > expected = {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};

        for (int f=0; f<layer->filters.size(); f++) {
            layer->filters[f]->bias = 1;
        }

        net->isTraining = false;
        layer->hasActivation = false;
        layer->forward();

        for (int f=0; f<layer->filters.size(); f++) {
            EXPECT_EQ( layer->filters[f]->activationMap, expected );
        }
    }

    // Runs the sumMap values through an activation function when provided (sigmoid, in this case)
    TEST_F(ConvForwardFixture, forward_7) {

        layer->hasActivation = true;
        layer->activationC = &NetMath::sigmoid<Filter>;
        std::vector<std::vector<double> > expected = {{0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049},{0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049},{0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049},{0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049},{0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049,0.7310585786300049}};

        for (int f=0; f<layer->filters.size(); f++) {
            layer->filters[f]->bias = 1;
        }

        net->isTraining = false;
        net->dropout = 1;
        layer->forward();

        for (int f=0; f<layer->filters.size(); f++) {
            EXPECT_EQ( layer->filters[f]->activationMap, expected );
        }
    }

    class ConvBackwardFixture : public ::testing::Test {
    public:
        virtual void SetUp () {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->l1 = 0;
            net->l2 = 0;
            net->miniBatchSize = 1;
            net->weightInitFn = &NetMath::uniform;

            prevLayer = new FCLayer(0, 75);
            nextLayerA = new FCLayer(0, 100);
            nextLayerB = new ConvLayer(0, 2);
            nextLayerB->filterSize = 5;
            nextLayerB->zeroPadding = 1;
            nextLayerB->channels = 4;
            nextLayerB->stride = 1;
            nextLayerB->outMapSize = 5;

            layer = new ConvLayer(0, 4);
            layer->filterSize = 3;
            layer->zeroPadding = 1;
            layer->channels = 3;
            layer->stride = 1;
            layer->outMapSize = 5;
            layer->inMapValuesCount = 25;

            layer->assignPrev(prevLayer);
            layer->assignNext(nextLayerB);
            nextLayerB->assignPrev(layer);
            prevLayer->init(0);
            layer->init(1);
            nextLayerB->init(2);

            for (int f=0; f<layer->filters.size(); f++) {
                layer->filters[f]->weights = {{{1,2,3},{4,5,6},{7,8,9}}, {{1,2,3},{4,5,6},{7,8,9}}, {{1,2,3},{4,5,6},{7,8,9}}};
                layer->filters[f]->sumMap = {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
                layer->filters[f]->errorMap = {{3,3,3,3,3},{3,3,3,3,3},{3,3,3,3,3},{3,3,3,3,3},{3,3,3,3,3}};
            }

            for (int f=0; f<nextLayerB->filters.size(); f++) {
                nextLayerB->filters[f]->weights = {{{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}},
                                                   {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}},
                                                   {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}},
                                                   {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}},
                                                   {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}}};
                nextLayerB->filters[f]->sumMap = {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
                nextLayerB->filters[f]->errorMap = {{3,3,3,3,3},{3,3,3,3,3},{3,3,3,3,3},{3,3,3,3,3},{3,3,3,3,3}};
            }

            prevLayer->actvns = {};
            for (int n=0; n<prevLayer->neurons.size(); n++) {
                prevLayer->actvns[n] = 0.5;
            }

            for (int f=0; f<layer->filters.size(); f++) {
                layer->filters[f]->dropoutMap = {{true,true,true,true,true},{true,true,true,true,true},{true,true,true,true,true},{true,true,true,true,true},{true,true,true,true,true}};
            }
        }

        virtual void TearDown () {
            delete prevLayer;
            delete layer;
            delete nextLayerA;
            delete nextLayerB;
            Network::deleteNetwork();
        }

        Network* net;
        FCLayer* prevLayer;
        FCLayer* nextLayerA;
        ConvLayer* nextLayerB;
        ConvLayer* layer;
    };

    // Calculates the error map correctly when the next layer is an FCLayer
    TEST(ConvLayer, backward_1) {

        Network::deleteNetwork();
        Network::newNetwork();
        Network* net = Network::getInstance(0);
        net->weightInitFn = &NetMath::uniform;
        net->weightsConfig["limit"] = 0.1;

        FCLayer* fcLayer = new FCLayer(0, 4);
        FCLayer* fcLayerPrev = new FCLayer(0, 18);
        ConvLayer* convLayer = new ConvLayer(0, 2);
        convLayer->channels = 1;
        convLayer->filterSize = 3;
        convLayer->stride = 1;
        convLayer->zeroPadding = 0;
        convLayer->outMapSize = 2;
        convLayer->inMapValuesCount = 4;

        convLayer->assignPrev(fcLayerPrev);
        convLayer->assignNext(fcLayer);
        fcLayer->assignPrev(convLayer);
        convLayer->hasActivation = false;

        fcLayerPrev->init(0);
        convLayer->init(1);
        fcLayer->init(2);

        fcLayer->errs = {};
        for (int n=0; n<fcLayer->neurons.size(); n++) {
            fcLayer->errs.push_back(((double)n+1)/5);
            fcLayer->weights[n] = {0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2};
        }

        convLayer->filters[0]->sumMap = {{0,0},{0,0}};
        convLayer->filters[1]->sumMap = {{0,0},{0,0}};
        convLayer->filters[0]->errorMap = {{0,0},{0,0}};
        convLayer->filters[1]->errorMap = {{0,0},{0,0}};
        convLayer->filters[0]->activationMap = {{0.1,0.2},{0.3,0.4}};
        convLayer->filters[1]->activationMap = {{0.5,0.6},{0.7,0.8}};

        std::vector<std::vector<std::vector<double> > > expected = { {{1.8, 1.6}, {1.4, 1.2}}, {{1, 0.8}, {0.6, 0.4}} };

        convLayer->backward();

        for (int f=0; f<convLayer->filters.size(); f++) {
            for (int r=0; r<convLayer->filters[f]->errorMap.size(); r++) {
                for (int c=0; c<convLayer->filters[f]->errorMap.size(); c++) {
                    EXPECT_NEAR( convLayer->filters[f]->errorMap[r][c], expected[f][r][c], 1e-8 );
                }
            }
        }
    }

    // Sets the errorMap values to 0 when dropped out
    TEST_F(ConvBackwardFixture, backward_2) {

        layer->outMapSize = 5;
        layer->backward();

        std::vector<std::vector<double> > expected = {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}};

        for (int f=0; f<layer->filters.size(); f++) {
            EXPECT_EQ( layer->filters[f]->errorMap, expected );
        }
    }

    // Does not increment the deltaBias when all values are dropped out
    TEST_F(ConvBackwardFixture, backward_3) {

        for (int f=0; f<layer->filters.size(); f++) {
            layer->filters[f]->deltaBias = f;
        }

        layer->outMapSize = 5;
        layer->backward();

        for (int f=0; f<layer->filters.size(); f++) {
            EXPECT_EQ( layer->filters[f]->deltaBias, f );
        }
    }

    // Does not increment the deltaWeights when all values are dropped out
    TEST_F(ConvBackwardFixture, backward_4) {

        std::vector<std::vector<std::vector<double> > > expected = {{{1,2,3},{4,5,6},{7,8,9}}, {{1,2,3},{4,5,6},{7,8,9}}, {{1,2,3},{4,5,6},{7,8,9}}};

        for (int f=0; f<layer->filters.size(); f++) {
            layer->filters[f]->deltaWeights = expected;
        }

        EXPECT_EQ( layer->stride, 1 );
        layer->backward();

        EXPECT_EQ( prevLayer->neurons.size(), 75 );

        for (int f=0; f<layer->filters.size(); f++) {
            EXPECT_EQ( layer->filters[f]->deltaWeights, expected );
        }
    }

    // Does otherwise change the deltaBias and deltaWeights values
    TEST_F(ConvBackwardFixture, backward_5) {

        std::vector<std::vector<std::vector<double> > > expected = {{{1,2,3},{4,5,6},{7,8,9}}, {{1,2,3},{4,5,6},{7,8,9}}, {{1,2,3},{4,5,6},{7,8,9}}};

        for (int f=0; f<layer->filters.size(); f++) {
            layer->filters[f]->dropoutMap = {{false,false,false,false,false},{false,false,false,false,false},{false,false,false,false,false},{false,false,false,false,false},{false,false,false,false,false}};
            layer->filters[f]->deltaBias = f;
            layer->filters[f]->deltaWeights = expected;
        }

        layer->outMapSize = 5;
        layer->backward();

        for (int f=0; f<layer->filters.size(); f++) {
            EXPECT_NE( layer->filters[f]->deltaBias, f );
            EXPECT_NE( layer->filters[f]->deltaWeights, expected );
        }
    }

    // Calculates errormap values correctly when the next layer is Conv
    TEST_F(ConvBackwardFixture, backward_6) {
        nextLayerB->filterSize = 3;
        nextLayerB->size = 1;
        nextLayerB->stride = 2;

        Filter* filter = new Filter();
        filter->weights = {{{-1, 0, -1}, {1, 0, 1}, {1, -1, 0}}, {{-1, 0, -1}, {1, 0, 1}, {1, -1, 0}}};
        filter->init(0);
        filter->errorMap = {{0.5, -0.2, 0.1}, {0, -0.4, -0.1}, {0.2, 0.6, 0.3}};

        nextLayerB->filters = {filter};
        layer->filters = {new Filter(), new Filter()};
        layer->filters[0]->weights = {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}};
        layer->filters[0]->init(0);
        layer->filters[0]->errorMap = {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}};
        layer->filters[1]->weights = {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}};
        layer->filters[1]->init(0);
        layer->filters[1]->errorMap = {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}};

        layer->backward();

        std::vector<std::vector<double> > expected = {{0,0.3,0,-0.1,0},{-0.5,0.2,0.2,0.6,-0.1},{0,-0.4,0,-0.5,0},{0,-1.2,0.4,-1,0.1},{0,0.8,0,0.9,0}};

        for (int r=0; r<5; r++) {
            for (int c=0; c<5; c++) {
                EXPECT_NEAR( layer->filters[0]->errorMap[r][c], expected[r][c], 1e-8 );
                EXPECT_NEAR( layer->filters[1]->errorMap[r][c], expected[r][c], 1e-8 );
            }
        }
        // delete filter;
    }

    // Maps the errors in the PoolLayer, 1 to 1, to the filters' errorMap values
    TEST(ConvLayer, backward_7) {

        Network::deleteNetwork();
        Network::newNetwork();
        Network* net = Network::getInstance(0);
        net->dropout = 1;
        net->weightInitFn = &NetMath::uniform;
        net->weightsConfig["limit"] = 0.1;

        FCLayer* prevLayer = new FCLayer(0, 75);

        ConvLayer* convLayer = new ConvLayer(0, 2);
        convLayer->channels = 1;
        convLayer->filterSize = 3;
        convLayer->stride = 1;
        convLayer->zeroPadding = 0;
        convLayer->outMapSize = 2;
        convLayer->inMapValuesCount = 4;
        convLayer->hasActivation = false;


        PoolLayer* poolLayer = new PoolLayer(0, 2);
        poolLayer->stride = 2;
        poolLayer->outMapSize = 1;
        poolLayer->inMapValuesCount = 8;

        prevLayer->assignNext(convLayer);
        convLayer->assignPrev(prevLayer);
        convLayer->assignNext(poolLayer);
        poolLayer->assignPrev(convLayer);

        prevLayer->init(0);
        convLayer->init(1);
        poolLayer->init(2);

        convLayer->filters[0]->sumMap = {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
        convLayer->filters[0]->errorMap = {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
        convLayer->filters[1]->sumMap = {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
        convLayer->filters[1]->errorMap = {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};

        poolLayer->errors = {{
            {1,2,3,4,5},
            {5,4,3,2,1},
            {1,2,3,4,5},
            {5,4,3,2,1},
            {1,2,3,4,5}
        }, {
            {6,7,8,8,9},
            {6,7,8,8,9},
            {0,9,8,7,6},
            {6,7,8,8,9},
            {0,9,8,7,6}
        }};

        prevLayer->actvns = {};
        for (int n=0; n<75; n++) {
            prevLayer->actvns[n] = 0.5;
        }

        std::vector<std::vector<double> > expected1 = {{1,2,3,4,5},{5,4,3,2,1},{1,2,3,4,5},{5,4,3,2,1},{1,2,3,4,5}};
        std::vector<std::vector<double> > expected2 = {{6,7,8,8,9},{6,7,8,8,9},{0,9,8,7,6},{6,7,8,8,9},{0,9,8,7,6}};

        EXPECT_EQ( prevLayer->sums.size(), 75 );

        convLayer->backward();

        EXPECT_EQ( convLayer->filters[0]->errorMap, expected1 );
        EXPECT_EQ( convLayer->filters[1]->errorMap, expected2 );

        delete prevLayer;
        delete convLayer;
        delete poolLayer;

        Network::deleteNetwork();
    }


    class ConvResetDeltaWFixture : public ::testing::Test {
    public:
        virtual void SetUp () {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);

            layer = new ConvLayer(0, 3);

            for (int f=0; f<3; f++) {
                layer->filters.push_back(new Filter());
                layer->filters[f]->errorMap = {{1,1,1},{1,1,1},{1,1,1}};
                layer->filters[f]->deltaWeights = {{{1,1,1},{1,1,1},{1,1,1}},{{1,1,1},{1,1,1},{1,1,1}}};
                layer->filters[f]->dropoutMap = {{true,true,true},{true,true,true},{true,true,true}};
            }

            layer2 = new ConvLayer(0, 5);

            for (int f=0; f<5; f++) {
                layer2->filters.push_back(new Filter());
                layer2->filters[f]->errorMap = {{1,1,1},{1,1,1},{1,1,1}};
                layer2->filters[f]->deltaWeights = {{{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}}};
                layer2->filters[f]->dropoutMap = {{true,true,true,true,true},{true,true,true,true,true},
                    {true,true,true,true,true},{true,true,true,true,true},{true,true,true,true,true}};
            }
        }

        virtual void TearDown () {
            delete layer;
            delete layer2;
            Network::deleteNetwork();
        }

        Network* net;
        ConvLayer* layer;
        ConvLayer* layer2;
    };

    // Sets all filters' deltaWeights values to 0
    TEST_F(ConvResetDeltaWFixture, resetDeltaWeights_1) {
        std::vector<std::vector<std::vector<double> > > expectedA = {{{0,0,0},{0,0,0},{0,0,0}},{{0,0,0},{0,0,0},{0,0,0}}};
        std::vector<std::vector<std::vector<double> > > expectedB = {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}};

        layer->resetDeltaWeights();
        layer2->resetDeltaWeights();


        for (int f=0; f<3; f++) {
            EXPECT_EQ( layer->filters[f]->deltaWeights, expectedA );
        }

        for (int f=0; f<5; f++) {
            EXPECT_EQ( layer2->filters[f]->deltaWeights, expectedB );
        }
    }

    // Sets all the filters' dropoutMap values to false
    TEST_F(ConvResetDeltaWFixture, resetDeltaWeights_2) {
        std::vector<std::vector<bool> > expectedA = {{false,false,false},{false,false,false},{false,false,false}};
        std::vector<std::vector<bool> > expectedB = {{false,false,false,false,false},{false,false,false,false,false},
                {false,false,false,false,false},{false,false,false,false,false},{false,false,false,false,false}};

        layer->resetDeltaWeights();
        layer2->resetDeltaWeights();

        for (int f=0; f<3; f++) {
            EXPECT_EQ( layer->filters[f]->dropoutMap, expectedA );
        }

        for (int f=0; f<5; f++) {
            EXPECT_EQ( layer2->filters[f]->dropoutMap, expectedB );
        }
    }

    // Sets the filters' deltaBias to 0
    TEST_F(ConvResetDeltaWFixture, resetDeltaWeights_3) {

        layer->resetDeltaWeights();
        layer2->resetDeltaWeights();

        for (int f=0; f<3; f++) {
            EXPECT_EQ( layer->filters[f]->deltaBias, 0 );
        }

        for (int f=0; f<5; f++) {
            EXPECT_EQ( layer2->filters[f]->deltaBias, 0 );
        }
    }

    TEST_F(ConvResetDeltaWFixture, resetDeltaWeights_4) {

        layer->resetDeltaWeights();
        layer2->resetDeltaWeights();

        std::vector<std::vector<double> > expected = {{0,0,0},{0,0,0},{0,0,0}};

        for (int f=0; f<3; f++) {
            EXPECT_EQ( layer->filters[f]->errorMap, expected );
        }

        for (int f=0; f<5; f++) {
            EXPECT_EQ( layer2->filters[f]->errorMap, expected );
        }
    }

    class ConvApplyDeltaWFixture : public ::testing::Test {
    public:
        virtual void SetUp () {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->learningRate = 1;
            net->updateFnIndex = 0;
            net->miniBatchSize = 1;
            net->l1 = 0;
            net->l2 = 0;

            layer = new ConvLayer(0, 5);

            for (int i=0; i<4; i++) {
                layer->filters.push_back(new Filter());
                layer->filters[i]->weights = {{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}},{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}}};
                layer->filters[i]->bias = 0.5;
                layer->filters[i]->deltaBias = 1;
                layer->filters[i]->deltaWeights = {{{1,1,1},{1,1,1},{1,1,1}},{{1,1,1},{1,1,1},{1,1,1}}};
            }
        }

        virtual void TearDown () {
            delete layer;
            Network::deleteNetwork();
        }

        Network* net;
        ConvLayer* layer;
    };

    // Increments the weights of all neurons with their respective deltas (when learning rate is 1)
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_1) {
        layer->applyDeltaWeights();

        std::vector<std::vector<std::vector<double> > > expected = {{{1.5,1.5,1.5},{1.5,1.5,1.5},{1.5,1.5,1.5}},{{1.5,1.5,1.5},{1.5,1.5,1.5},{1.5,1.5,1.5}}};
        EXPECT_EQ( layer->filters[0]->weights, expected );
        EXPECT_EQ( layer->filters[1]->weights, expected );
        EXPECT_EQ( layer->filters[2]->weights, expected );
        EXPECT_EQ( layer->filters[3]->weights, expected );
    }

    // Increments the bias of all filters with their deltaBias
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_2) {
        layer->applyDeltaWeights();
        EXPECT_EQ( layer->filters[0]->bias, 1.5 );
    }

    // Increments the net.l2Error by each weight, applied to the L2 formula
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_3) {
        net->l2 = 0.001;
        net->l2Error = 0;
        layer->applyDeltaWeights();
        EXPECT_DOUBLE_EQ( net->l2Error, 0.009 );
    }

    // Increments the net.l1Error by each weight, applied to the L1 formula
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_4) {
        net->l1 = 0.005;
        net->l1Error = 0;
        layer->applyDeltaWeights();
        EXPECT_DOUBLE_EQ( net->l1Error, 0.18 );
    }

    // Increments the net.maxNormTotal if the net.maxNorm is configured (then sets it to 0)
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_5) {
        net->maxNorm = 3;
        net->maxNormTotal = 10;
        layer->applyDeltaWeights();
        EXPECT_EQ( net->maxNormTotal, 0 );
    }

    // Increments the bias of all filters with their deltaBias, following the gain function
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_6) {

        net->updateFnIndex = 1;

        for (int f=0; f<4; f++) {
            layer->filters[f]->biasGain = 0.5;
            layer->filters[f]->weightGain = {{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}},{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}}};
        }

        layer->applyDeltaWeights();
        layer->applyDeltaWeights();

        for (int f=0; f<4; f++) {
            EXPECT_EQ( layer->filters[f]->bias, 1.55 );
        }
    }

    // Increments the bias of all filters with their deltaBias, following the adagrad function
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_7) {

        net->updateFnIndex = 2;

        for (int f=0; f<4; f++) {
            layer->filters[f]->biasCache = 0.5;
            layer->filters[f]->weightsCache = {{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}},{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}}};
        }

        layer->applyDeltaWeights();

        for (int f=0; f<4; f++) {
            EXPECT_NEAR( layer->filters[f]->bias, 1.3165, 1e-3 );
        }
    }

    // Increments the bias of all filters with their deltaBias, following the rmsprop function
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_8) {

        net->updateFnIndex = 3;
        net->rmsDecay = 0.99;

        for (int f=0; f<4; f++) {
            layer->filters[f]->biasCache = 0.5;
            layer->filters[f]->weightsCache = {{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}},{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}}};
        }

        layer->applyDeltaWeights();

        for (int f=0; f<4; f++) {
            EXPECT_NEAR( layer->filters[f]->bias, 1.90719, 1e-3 );
        }
    }

    // Increments the bias of all filters with their deltaBias, following the adam function
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_9) {

        net->updateFnIndex = 4;

        for (int f=0; f<4; f++) {
            layer->filters[f]->biasCache = 0.5;
            layer->filters[f]->m = 0;
            layer->filters[f]->v = 0;
        }

        layer->applyDeltaWeights();

        for (int f=0; f<4; f++) {
            EXPECT_NEAR( layer->filters[f]->bias, 2.49319, 1e-3 );
        }
    }

    // Increments the bias of all filters with their deltaBias, following the adadelta function
    TEST_F(ConvApplyDeltaWFixture, applyDeltaWeights_10) {

        net->updateFnIndex = 5;
        net->rho = 0.95;

        for (int f=0; f<4; f++) {
            layer->filters[f]->biasCache = 0.5;
            layer->filters[f]->adadeltaBiasCache = 0.25;

            layer->filters[f]->weightsCache = {{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}},{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}}};
            layer->filters[f]->adadeltaCache = {{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}},{{0.5,0.5,0.5},{0.5,0.5,0.5},{0.5,0.5,0.5}}};
        }

        layer->applyDeltaWeights();

        for (int f=0; f<4; f++) {
            EXPECT_NEAR( layer->filters[f]->bias, 1.19007, 1e-3 );
        }
    }
}

namespace PoolLayer_cpp {

    // Assigns the type as Pool
    TEST(PoolLayer, constructor) {
        PoolLayer* layer = new PoolLayer(0, 1);
        EXPECT_EQ( layer->type, "Pool" );
        delete layer;
    }

    // Assigns the given layer pointer to this layer's nextLayer
    TEST(PoolLayer, assignNext) {
        PoolLayer* l1 = new PoolLayer(0, 1);
        PoolLayer* l2 = new PoolLayer(0, 1);

        l1->assignNext(l2);
        EXPECT_EQ( l1->nextLayer, l2 );

        delete l1;
        delete l2;
    }

    // Assigns the given layer pointer to this layer's prevLayer
    TEST(PoolLayer, assignPrev) {
        PoolLayer* l1 = new PoolLayer(0, 1);
        PoolLayer* l2 = new PoolLayer(0, 1);

        l2->assignPrev(l1);
        EXPECT_EQ( l2->prevLayer, l1 );

        delete l1;
        delete l2;
    }

    // Inits an activations volume with dimensions channels x outMapSize x outMapSize, with 0 values
    TEST(PoolLayer, init_1) {
        PoolLayer* layer = new PoolLayer(0, 1);
        layer->channels = 3;
        layer->outMapSize = 2;
        layer->inMapValuesCount = 9;

        std::vector<std::vector<std::vector<double> > > expected = { {{0,0},{0,0}}, {{0,0},{0,0}}, {{0,0},{0,0}} };
        layer->init(0);

        EXPECT_EQ( layer->activations, expected );
        delete layer;
    }

    // Inits an errors volume with dimensions channels x prevLayerOutWidth x prevLayerOutWidth, with 0 values
    TEST(PoolLayer, init_2) {
        PoolLayer* layer = new PoolLayer(0, 1);
        layer->channels = 2;
        layer->outMapSize = 2;
        layer->inMapValuesCount = 9;

        std::vector<std::vector<std::vector<double> > > expected = { {{0,0,0},{0,0,0},{0,0,0}}, {{0,0,0},{0,0,0},{0,0,0}} };
        layer->init(0);

        EXPECT_EQ( layer->errors, expected );
        delete layer;
    }

    // Inits an indeces volume with the dimensions as the activations volume, with {0,0} values
    TEST(PoolLayer, init_3) {
        PoolLayer* layer = new PoolLayer(0, 1);
        layer->channels = 1;
        layer->outMapSize = 2;
        layer->inMapValuesCount = 4;

        std::vector<std::vector<std::vector<std::vector<int> > > > expected = { { {{0,0},{0,0}}, {{0,0},{0,0}} } };
        layer->init(0);
        EXPECT_EQ( layer->indeces, expected );

        delete layer;
    }

    // Sets the layer->prevLayerOutWidth to the square root of the inMapValuesCount
    TEST(PoolLayer, init_4) {
        PoolLayer* layer = new PoolLayer(0, 1);
        layer->channels = 1;
        layer->outMapSize = 2;

        layer->inMapValuesCount = 4;
        layer->init(0);
        EXPECT_EQ( layer->prevLayerOutWidth, 2 );

        layer->inMapValuesCount = 9;
        layer->init(0);
        EXPECT_EQ( layer->prevLayerOutWidth, 3 );

        delete layer;
    }


    // Does not apply activation function if not provided
    TEST(PoolLayer, forward_1) {
        Network::deleteNetwork();
        Network::newNetwork();
        PoolLayer* layer = new PoolLayer(0, 2);
        FCLayer* prevLayer = new FCLayer(0, 18);
        layer->hasActivation = false;
        layer->channels = 2;
        layer->stride = 2;
        layer->outMapSize = 2;
        layer->inMapValuesCount = 9;
        layer->prevLayerOutWidth = 9;
        layer->activations = { {{1,2},{3,4}}, {{5,6},{7,8}} };
        prevLayer->assignNext(layer);
        layer->assignPrev(prevLayer);
        prevLayer->init(0);
        prevLayer->biases = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        layer->init(1);

        prevLayer->actvns = {};
        for (int i=0; i<prevLayer->size; i++) {
            prevLayer->actvns[i] = i%9;
        }

        layer->forward();

        std::vector<std::vector<std::vector<double> > > expected = { {{4,6},{7,8}}, {{4,6},{7,8}} };
        EXPECT_EQ( layer->activations, expected );

        delete layer;
        delete prevLayer;
    }

    // Applies activation function if provided (using sigmoid for test)
    TEST(PoolLayer, forward_2) {
        Network::deleteNetwork();
        Network::newNetwork();
        PoolLayer* layer = new PoolLayer(0, 2);
        FCLayer* prevLayer = new FCLayer(0, 18);
        layer->hasActivation = true;
        layer->activationP = &NetMath::sigmoid<Network>;
        layer->channels = 2;
        layer->stride = 2;
        layer->outMapSize = 2;
        layer->inMapValuesCount = 9;
        layer->prevLayerOutWidth = 9;
        layer->activations = { {{1,2},{3,4}}, {{5,6},{7,8}} };

        layer->assignPrev(prevLayer);
        prevLayer->init(0);
        layer->init(1);

        prevLayer->actvns = {};
        for (int i=0; i<prevLayer->size; i++) {
            prevLayer->actvns[i] = i%9;
        }

        layer->forward();

        std::vector<std::vector<std::vector<double> > > expected = { {{0.982014,0.997527},{0.999089,0.999665}}, {{0.982014,0.997527},{0.999089,0.999665}} };
        for (int r=0; r<2; r++) {
            for (int c=0; c<2; c++) {
                EXPECT_NEAR( layer->activations[0][r][c], expected[0][r][c], 1e-3 );
                EXPECT_NEAR( layer->activations[1][r][c], expected[1][r][c], 1e-3 );
            }
        }
        delete layer;
        delete prevLayer;
    }


    class PoolBackwardFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->weightInitFn = &NetMath::uniform;

            poolLayer = new PoolLayer(0, 2);
            poolLayer->size = 2;
            poolLayer->stride = 2;
            poolLayer->channels = 1;
            poolLayer->outMapSize = 6;
            poolLayer->inMapValuesCount = 144;
            poolLayer->prevLayerOutWidth = 12;
            poolLayer->hasActivation = false;

            convLayer = new ConvLayer(0, 1);
            convLayer->filterSize = 3;
            convLayer->zeroPadding = 1;
            convLayer->stride = 1;

            fcLayer = new FCLayer(0, 36);
            fcLayer->size = 36;
        }

        virtual void TearDown() {
            delete poolLayer;
            delete convLayer;
            delete fcLayer;
            Network::deleteNetwork();
        }

        Network* net;
        PoolLayer* poolLayer;
        ConvLayer* convLayer;
        FCLayer* fcLayer;
    };


    // Creates the error map correctly when the next layer is an FCLayer
    TEST_F(PoolBackwardFixture, backward_1) {

        PoolLayer* layer = new PoolLayer(0, 2);
        layer->stride = 2;
        layer->channels = 1;
        layer->outMapSize = 6;
        layer->inMapValuesCount = 36;
        layer->hasActivation = false;

        layer->assignNext(fcLayer);
        fcLayer->assignPrev(layer);
        layer->init(0);
        fcLayer->init(1);

        layer->errors = {{
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0}
        }};

        layer->indeces = {{
            {{0,1},{0,1},{0,1},{0,1},{0,1},{0,1}},
            {{1,1},{1,1},{0,1},{1,1},{1,1},{1,1}},
            {{1,1},{1,1},{1,1},{1,1},{1,1},{1,1}},
            {{0,1},{0,1},{0,1},{0,1},{0,1},{0,1}},
            {{1,1},{1,1},{0,1},{1,1},{1,1},{1,1}},
            {{1,1},{1,1},{1,1},{1,1},{1,1},{1,1}}
        }};

        fcLayer->errs = {};

        for (double n=0; n<fcLayer->size; n++) {
            fcLayer->errs.push_back(n ? n / 100 : 0);
            fcLayer->weights[n] = {0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5};
        }


        layer->backward();

        std::vector<std::vector<std::vector<double> > > expected = {{
            {0,0, 0,6.3, 0,12.6, 0,18.9, 0,25.2, 0,31.5},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,0, 0,0, 0,50.4, 0,0, 0,0, 0,0},
            {0,37.8, 0,44.1, 0,0, 0,56.7, 0,0, 0,6.3},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,12.6, 0,18.9, 0,25.2, 0,31.5, 0,37.8, 0,44.1},
            {0,50.4, 0,56.7, 0,0, 0,6.3, 0,12.6, 0,18.9},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,0, 0,0, 0,37.8, 0,0, 0,0, 0,0},
            {0,25.2, 0,31.5, 0,0, 0,44.1, 0,50.4, 0,56.7},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,0, 0,6.3, 0,12.6, 0,18.9, 0,25.2, 0,31.5}
        }};

        for (int i=0; i<12; i++) {
            EXPECT_EQ( layer->errors[0][i].size(), expected[0][i].size() );
            for (int j=0; j<12; j++) {
                EXPECT_DOUBLE_EQ( layer->errors[0][i][j], expected[0][i][j] );
            }
        }
    }

    // Creates the error map correctly when the next layer is a ConvLayer
    TEST_F(PoolBackwardFixture, backward_2) {

        PoolLayer* layer = new PoolLayer(0, 3);
        layer->channels = 1;
        layer->stride = 2;
        layer->outMapSize = 6;
        layer->inMapValuesCount = 169;

        layer->assignNext(convLayer);
        layer->init(0);
        convLayer->init(1);

        convLayer->filters.push_back(new Filter());

        convLayer->filters[0]->errorMap = {
            {1,2,3,4,5,6},
            {7,4,7,2,9,2},
            {1,9,3,7,3,6},
            {2,5,2,6,8,3},
            {8,2,4,9,2,7},
            {1,7,3,7,3,5}
        };
        convLayer->filters[0]->weights = {{
            {1,2,3},
            {4,5,2},
            {3,2,1}
        }};

        layer->errors = {{
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0,0}
        }};

        layer->indeces = {{
            {{0,1},{2,1},{0,2},{2,2},{0,1},{0,0}},
            {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}},
            {{0,1},{2,1},{0,2},{2,2},{0,1},{0,0}},
            {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}},
            {{0,1},{0,2},{1,1},{2,0},{0,2},{2,1}},
            {{1,1},{0,0},{2,2},{1,1},{0,0},{1,1}}
        }};

        layer->backward();

        std::vector<std::vector<std::vector<double> > > expected = {{
            {0, 31,  0,  0,  0,  0, 63,  0,  0, 83, 71,  0,  0},
            {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
            {70,  0,100, 60,111,  0,112,  0,202,  0, 66,  0,  0},
            {0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
            {0, 76,  0,  0,  0,  0,110,  0,  0,116, 79,  0,  0},
            {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
            {77,  0, 97,113,103,  0,124,  0,250,  0, 66,  0,  0},
            {0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
            {0, 76,  0,  0, 80,  0,  0,  0,  0,  0,119,  0,  0},
            {0, 0,  0,  0,  0,121,  0,  0,  0,  0,  0,  0,  0},
            {0,  0, 73,  0,  0,  0,125,  0, 83,  0,  0, 72,  0},
            {0, 55,  0,  0,  0,  0,  0, 81,  0,  0,  0, 47,  0},
            {0,  0,  0,  0,  0,  0, 94,  0,  0,  0,  0,  0,  0}
        }};

        EXPECT_EQ( layer->errors[0][0].size(), 13 );
        EXPECT_EQ( expected[0][0].size(), 13 );
        EXPECT_EQ( layer->errors, expected );
    }

    // Creates the error map correctly when the next layer is a PoolLayer
    TEST_F(PoolBackwardFixture, backward_3) {

        PoolLayer* layer = new PoolLayer(0, 3);
        layer->stride = 3;
        layer->channels = 2;
        layer->outMapSize = 3;
        layer->inMapValuesCount = 81;
        layer->hasActivation = false;

        layer->assignNext(poolLayer);
        layer->init(0);
        poolLayer->init(1);

        layer->errors = {{
            {0,0,0, 0,0,0, 0,0,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},

            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},

            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},
        }, {
            {0,0,0, 0,0,0, 0,0,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},

            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},

            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},
        }};

        layer->indeces = {{
            {{2,2},{0,1},{1,1}},
            {{0,1},{2,0},{0,2}},
            {{0,1},{0,1},{0,1}}
        }, {
            {{0,0},{0,0},{0,0}},
            {{1,2},{2,1},{0,1}},
            {{0,0},{0,2},{0,1}}
        }};

        poolLayer->errors = {{
            {1,2,3},
            {4,5,6},
            {7,8,9}
        }, {
            {5,9,3},
            {8,2,4},
            {6,7,1}
        }};

        layer->backward();

        std::vector<std::vector<std::vector<double> > > expected = {{
            {0,0,0,0,2,0,0,0,0},
            {0,0,0,0,0,0,0,3,0},
            {0,0,1,0,0,0,0,0,0},
            {0,4,0,0,0,0,0,0,6},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,5,0,0,0,0,0},
            {0,7,0,0,8,0,0,9,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0}
        }, {
            {5,0,0,9,0,0,3,0,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,4,0},
            {0,0,8,0,0,0,0,0,0},
            {0,0,0,0,2,0,0,0,0},
            {6,0,0,0,0,7,0,1,0},
            {0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0}
        }};

        EXPECT_EQ( layer->errors, expected );
    }

    // Applies activation derivative when the activation is sigmoid
    TEST_F(PoolBackwardFixture, backward_4) {

        Network::getInstance(0)->weightsConfig["limit"] = 0.1;
        PoolLayer* layer = new PoolLayer(0, 2);
        layer->stride = 2;
        layer->channels = 1;
        layer->outMapSize = 6;
        layer->inMapValuesCount = 36;
        layer->hasActivation = true;
        layer->activationP = &NetMath::sigmoid;

        layer->assignNext(fcLayer);
        fcLayer->assignPrev(layer);
        layer->init(0);
        fcLayer->init(1);

        fcLayer->errs = {};
        for (double n=0; n<fcLayer->size; n++) {
            fcLayer->errs.push_back(n ? n / 100 : 0);
            fcLayer->weights[n] = {0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5};
        }

        layer->errors = {{
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0,0},
        }};
        layer->indeces = {{
            {{0,1},{0,1},{0,1},{0,1},{0,1},{0,1}},
            {{1,1},{1,1},{0,1},{1,1},{1,1},{1,1}},
            {{1,1},{1,1},{1,1},{1,1},{1,1},{1,1}},
            {{0,1},{0,1},{0,1},{0,1},{0,1},{0,1}},
            {{1,1},{1,1},{0,1},{1,1},{1,1},{1,1}},
            {{1,1},{1,1},{1,1},{1,1},{1,1},{1,1}}
        }};

        layer->backward();
        std::vector<std::vector<std::vector<double> > > expected = {{
            {0,0, 0, 0.011526349447153203, 0,0.000042487105415310055, 0,1.1702970200399024e-7, 0,2.865355952475672e-10, 0,6.57474075183004e-13},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0.011526349447153203},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,0.000042487105415310055, 0,1.1702970200399024e-7, 0,2.865355952475672e-10, 0,6.57474075183004e-13, 0,0, 0,0},
            {0,0, 0,0, 0,0, 0,0.011526349447153203, 0,0.000042487105415310055, 0,1.1702970200399024e-7},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,2.865355952475672e-10, 0,6.57474075183004e-13, 0,0, 0,0, 0,0, 0,0},
            {0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
            {0,0, 0,0.011526349447153203, 0,0.000042487105415310055, 0,1.1702970200399024e-7, 0,2.865355952475672e-10, 0,6.57474075183004e-13}
        }};

        EXPECT_EQ( fcLayer->weights[0].size(), 36 );

        for (int i=0; i<12; i++) {
            EXPECT_EQ( layer->errors[0][i].size(), expected[0][i].size() );
            for (int j=0; j<12; j++) {
                EXPECT_NEAR( layer->errors[0][i][j], expected[0][i][j], 1e-8 );
            }
        }
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

    // Sets the neuron deltaBias to 0
    TEST_F(NeuronInitFixture, init_2) {
        testN->deltaBias = 999;
        testN->init(0, 5);
        EXPECT_EQ( testN->deltaBias, 0 );
    }

    // Sets the neuron biasGain to 1 if the net's updateFn is gain
    TEST_F(NeuronInitFixture, init_3) {
        net->updateFnIndex = 1;
        testN->init(0, 5);
        EXPECT_EQ( testN->biasGain, 1 );
    }

    // Sets the neuron weightGain to a vector of 1s, with the same size as the weights vector when updateFn is gain
    TEST_F(NeuronInitFixture, init_4) {
        net->updateFnIndex = 1;
        testN->init(0, 5);
        std::vector<double> expected = {1,1,1,1,1};
        EXPECT_EQ( testN->weightGain, expected );
    }

    // Does not set the biasGain or weightGain to anything if updateFn is not gain
    TEST_F(NeuronInitFixture, init_5) {
        net->updateFnIndex = 2;
        testN->init(0, 5);
        EXPECT_EQ( testN->weightGain.size(), 0 );
    }

    // Sets the neuron biasCache to 0 if the updateFn is adagrad
    TEST_F(NeuronInitFixture, init_6) {
        net->updateFnIndex = 2;
        testN->biasCache = 1;
        testN->init(0, 5);
        EXPECT_EQ( testN->biasCache, 0 );
    }

    // Sets the neuron weightsCache to a vector of zeroes with the same size as the weights when updateFn is adagrad
    TEST_F(NeuronInitFixture, init_7) {
        net->updateFnIndex = 2;
        testN->init(0, 5);
        std::vector<double> expected = {0,0,0,0,0};
        EXPECT_EQ( testN->weightsCache, expected );
    }

    // Does not set the biasCache or weightsCache to anything if updateFn is not adagrad
    TEST_F(NeuronInitFixture, init_8) {
        net->updateFnIndex = 1;
        testN->biasCache = 12234;
        testN->init(0, 5);
        EXPECT_EQ( testN->biasCache, 12234 );
        EXPECT_EQ( testN->weightsCache.size(), 0 );
    }

    // Sets the neuron biasCache to 0 if the updateFn is rmsprop
    TEST_F(NeuronInitFixture, init_9) {
        net->updateFnIndex = 3;
        testN->biasCache = 1;
        testN->init(0, 5);
        EXPECT_EQ( testN->biasCache, 0 );
    }

    // Sets the neuron weightsCache to a vector of zeroes with the same size as the weights when updateFn is rmsprop
    TEST_F(NeuronInitFixture, init_10) {
        net->updateFnIndex = 3;
        testN->init(0, 5);
        std::vector<double> expected = {0,0,0,0,0};
        EXPECT_EQ( testN->weightsCache, expected );
    }

    // Sets the neuron m and neuron v to 0 if the updateFn is adam
    TEST_F(NeuronInitFixture, init_11) {
        net->updateFnIndex = 4;
        testN->m = 1;
        testN->v = 1;
        testN->init(0, 5);
        EXPECT_EQ( testN->m, 0 );
        EXPECT_EQ( testN->v, 0 );
    }

    // Does not set the neuron m and neuron v to 0 if the updateFn is not adam
    TEST_F(NeuronInitFixture, init_12) {
        net->updateFnIndex = 3;
        testN->m = 1;
        testN->v = 1;
        testN->init(0, 5);
        EXPECT_EQ( testN->m, 1 );
        EXPECT_EQ( testN->v, 1 );
    }

    // Sets the neuron biasCache and adadeltaBiasCache to 0 if the updateFn is adadelta
    TEST_F(NeuronInitFixture, init_13) {
        net->updateFnIndex = 5;
        testN->biasCache = 1;
        testN->adadeltaBiasCache = 1;
        testN->init(0, 5);
        EXPECT_EQ( testN->biasCache, 0 );
        EXPECT_EQ( testN->adadeltaBiasCache, 0 );
    }

    // Sets the neuron weightsCache and adadeltaCache to a vector of zeroes with the same size as the weights when updateFn is adadelta
    TEST_F(NeuronInitFixture, init_14) {
        net->updateFnIndex = 5;
        testN->init(0, 5);
        std::vector<double> expected = {0,0,0,0,0};
        EXPECT_EQ( testN->weightsCache, expected );
        EXPECT_EQ( testN->adadeltaCache, expected );
    }

    // Does not set the biasCache or weightsCache to anything if updateFn is not adadelta
    TEST_F(NeuronInitFixture, init_15) {
        net->updateFnIndex = 1;
        testN->biasCache = 12234;
        testN->adadeltaBiasCache = 12234;
        testN->init(0, 5);
        EXPECT_EQ( testN->biasCache, 12234 );
        EXPECT_EQ( testN->adadeltaBiasCache, 12234 );
        EXPECT_EQ( testN->weightsCache.size(), 0 );
        EXPECT_EQ( testN->adadeltaCache.size(), 0 );
    }

    // Sets the network eluAlpha to the neuron, if the activation function is elu
    TEST_F(NeuronInitFixture, init_16) {
        net->activation = &NetMath::lrelu;
        net->lreluSlope = 0.1;
        testN->init(0, 5);
        EXPECT_NEAR(testN->lreluSlope, 0.1, 1e-6 );
    }

    // Sets the neuron rreluSlope to a number if the activation is rrelu
    TEST_F(NeuronInitFixture, init_17) {
        net->activation = &NetMath::rrelu;
        testN->rreluSlope = 0.1;
        testN->init(0, 5);
        EXPECT_NE( testN->rreluSlope, 0 );
        EXPECT_NE( testN->rreluSlope, 0.1 );
        EXPECT_GE( testN->rreluSlope, -0.1);
        EXPECT_LE( testN->rreluSlope, 0.1);
    }

    // Sets the network eluAlpha to the neuron, if the activation function is elu
    TEST_F(NeuronInitFixture, init_18) {
        net->activation = &NetMath::elu;
        net->eluAlpha = 0.1;
        testN->init(0, 5);
        EXPECT_NEAR(testN->eluAlpha, 0.1, 1e-6 );
    }
}

namespace Filter_cpp {

    class FilterInitFixture : public ::testing::Test {
    public:
        virtual void SetUp () {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            testFilter = new Filter();
            testFilter->weights = {{{1,2,3},{4,5,6},{7,8,9}}, {{1,2,3},{4,5,6},{7,8,9}}};
        }

        virtual void TearDown () {
            delete testFilter;
            Network::deleteNetwork();
        }

        Network* net;
        Filter* testFilter;
    };

    // Creates a volume of delta weights with depth==channels and the same spacial dimensions as the weights map, with 0 values
    TEST_F(FilterInitFixture, init_1) {
        testFilter->init(0);
        std::vector<std::vector<std::vector<double> > > expected = {{{0,0,0},{0,0,0},{0,0,0}}, {{0,0,0},{0,0,0},{0,0,0}}};
        EXPECT_EQ( testFilter->deltaWeights,  expected);
    }

    // Sets the filter.deltaBias value to 0
    TEST_F(FilterInitFixture, init_2) {
        testFilter->deltaBias = 99;
        testFilter->init(0);
        EXPECT_EQ( testFilter->deltaBias, 0 );
    }

    // Creates a weightGain map if the updateFn parameter is gain, with the same dimensions as weights, with 1 values
    TEST_F(FilterInitFixture, init_3) {
        net->updateFnIndex = 1;
        testFilter->init(0);
        std::vector<std::vector<std::vector<double> > > expected = {{{1,1,1},{1,1,1},{1,1,1}}, {{1,1,1},{1,1,1},{1,1,1}}};
        EXPECT_EQ( testFilter->weightGain,  expected);
    }

    // Creates a biasGain value of 1 if the updateFn parameter is gain
    TEST_F(FilterInitFixture, init_4) {
        testFilter->biasGain = 0;
        net->updateFnIndex = 1;
        testFilter->init(0);
        EXPECT_EQ( testFilter->biasGain, 1 );
    }

    // Does not create the weightGains and biasGain when the updateFn is not gain
    TEST_F(FilterInitFixture, init_5) {
        net->updateFnIndex = 99;
        testFilter->biasGain = 123;
        testFilter->init(0);
        EXPECT_EQ( testFilter->biasGain, 123 );
        EXPECT_EQ( testFilter->weightGain.size(), 0 );
    }

    // Creates a weightsCache map, with same dimensions as weights, with 0 values, and biasCache value of 0, if the updateFn is adagrad
    TEST_F(FilterInitFixture, init_6) {
        testFilter->biasCache = 123;
        net->updateFnIndex = 2;
        testFilter->init(0);
        std::vector<std::vector<std::vector<double> > > expected = {{{0,0,0},{0,0,0},{0,0,0}}, {{0,0,0},{0,0,0},{0,0,0}}};
        EXPECT_EQ( testFilter->weightsCache,  expected);
        EXPECT_EQ( testFilter->biasCache, 0 );
    }

    // Creates a weightsCache map, with same dimensions as weights, with 0 values, and biasCache value of 0, if the updateFn is rmsprop
    TEST_F(FilterInitFixture, init_7) {
        testFilter->biasCache = 123;
        net->updateFnIndex = 2;
        testFilter->init(0);
        std::vector<std::vector<std::vector<double> > > expected = {{{0,0,0},{0,0,0},{0,0,0}}, {{0,0,0},{0,0,0},{0,0,0}}};
        EXPECT_EQ( testFilter->weightsCache,  expected);
        EXPECT_EQ( testFilter->biasCache, 0 );
    }

    // Creates a weightsCache map, with same dimensions as weights, with 0 values, and biasCache value of 0, if the updateFn is adadelta
    TEST_F(FilterInitFixture, init_8) {
        testFilter->biasCache = 123;
        net->updateFnIndex = 2;
        testFilter->init(0);
        std::vector<std::vector<std::vector<double> > > expected = {{{0,0,0},{0,0,0},{0,0,0}}, {{0,0,0},{0,0,0},{0,0,0}}};
        EXPECT_EQ( testFilter->weightsCache,  expected);
        EXPECT_EQ( testFilter->biasCache, 0 );
    }

    // Does not create the weightsCache or biasCache if the updateFn is something else
    TEST_F(FilterInitFixture, init_9) {
        testFilter->biasCache = 123;
        net->updateFnIndex = 99;
        testFilter->init(0);
        EXPECT_EQ( testFilter->weightsCache.size(), 0 );
        EXPECT_EQ( testFilter->biasCache, 123 );
    }

    // Creates a adadeltaCache map, with same dimensions as weights, with 0 values, and adadeltaBiasCache value of 0, if the updateFn is adadelta
    TEST_F(FilterInitFixture, init_10) {
        net->updateFnIndex = 5;
        testFilter->adadeltaBiasCache = 123;
        testFilter->init(0);
        std::vector<std::vector<std::vector<double> > > expected = {{{0,0,0},{0,0,0},{0,0,0}}, {{0,0,0},{0,0,0},{0,0,0}}};
        EXPECT_EQ( testFilter->adadeltaBiasCache, 0 );
        EXPECT_EQ( testFilter->adadeltaCache, expected );
    }

    // Does not create adadeltaBiasCache or adadeltaCache when the updateFn is adagrad or rmsprop
    TEST_F(FilterInitFixture, init_11a) {
        net->updateFnIndex = 2;
        testFilter->adadeltaBiasCache = 123;
        testFilter->init(0);
        EXPECT_EQ( testFilter->adadeltaBiasCache, 123 );
        EXPECT_EQ( testFilter->adadeltaCache.size(), 0 );
    }
    TEST_F(FilterInitFixture, init_11b) {
        net->updateFnIndex = 3;
        testFilter->adadeltaBiasCache = 123;
        testFilter->init(0);
        EXPECT_EQ( testFilter->adadeltaBiasCache, 123 );
        EXPECT_EQ( testFilter->adadeltaCache.size(), 0 );
    }

    // Creates and sets filter.m and filter.v to 0 if the updateFn parameter is adam
    TEST_F(FilterInitFixture, init_12) {
        net->updateFnIndex = 4;
        testFilter->m = 99;
        testFilter->v = 99;
        testFilter->init(0);
        EXPECT_EQ( testFilter->m, 0 );
        EXPECT_EQ( testFilter->v, 0 );
    }

    // It does not create them if the updateFn is not adam
    TEST_F(FilterInitFixture, init_13) {
        net->updateFnIndex = 123;
        testFilter->m = 99;
        testFilter->v = 99;
        testFilter->init(0);
        EXPECT_EQ( testFilter->m, 99 );
        EXPECT_EQ( testFilter->v, 99 );
    }

    // Sets the filter.lreluSlope to the given value, if given a value
    TEST_F(FilterInitFixture, init_14) {
        net->activation = &NetMath::lrelu;
        net->lreluSlope = 123;
        testFilter->lreluSlope = 0;
        testFilter->init(0);
        EXPECT_EQ( testFilter->lreluSlope, 123 );
    }

    // Creates a random filter.rreluSlope number if the activation is rrelu
    TEST_F(FilterInitFixture, init_15) {
        net->activation = &NetMath::rrelu;
        testFilter->init(0);
        EXPECT_NE( testFilter->rreluSlope, 0 );
        EXPECT_NE( testFilter->rreluSlope, 0.1 );
        EXPECT_GE( testFilter->rreluSlope, -0.1);
        EXPECT_LE( testFilter->rreluSlope, 0.1);
    }

    // Sets the filter.eluAlpha to the given value, if given a value
    TEST_F(FilterInitFixture, init_16) {
        net->activation = &NetMath::elu;
        net->eluAlpha = 123;
        testFilter->eluAlpha = 0;
        testFilter->init(0);
        EXPECT_EQ( testFilter->eluAlpha, 123 );
    }
}

namespace NetMath_cpp {

    TEST(NetMath, sigmoid) {
        Neuron* testN = new Neuron();
        EXPECT_EQ( NetMath::sigmoid(1.681241237, false, testN), 0.8430688214048092 );
        EXPECT_EQ( NetMath::sigmoid(0.8430688214048092, true, testN), 0.21035474941074114 );
        delete testN;
    }

    TEST(NetMath, sigmoid_filter) {
        Filter* testF = new Filter();
        EXPECT_EQ( NetMath::sigmoid(1.681241237, false, testF), 0.8430688214048092 );
        EXPECT_EQ( NetMath::sigmoid(0.8430688214048092, true, testF), 0.21035474941074114 );
        delete testF;
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

    TEST(NetMath, tanh_filter) {
        Filter* testF = new Filter();
        EXPECT_EQ( NetMath::tanh(1, false, testF), 0.7615941559557649 );
        EXPECT_EQ( NetMath::tanh(0.5, false, testF), 0.46211715726000974);
        EXPECT_EQ( NetMath::tanh(0.5, true, testF), 0.7864477329659275 );
        EXPECT_EQ( NetMath::tanh(1.5, true, testF), 0.18070663892364855 );
        EXPECT_NE( NetMath::tanh(900, true, testF), NAN );
        delete testF;
    }

    TEST(NetMath, lecuntanh) {
        Neuron* testN = new Neuron();
        EXPECT_EQ( NetMath::lecuntanh(2.0, false, testN), 1.4929388053842507 );
        EXPECT_EQ( NetMath::lecuntanh(-2.0, false, testN), -1.4929388053842507 );
        EXPECT_EQ( NetMath::lecuntanh(2.0, true, testN), 0.2802507761872869 );
        EXPECT_EQ( NetMath::lecuntanh(-2.0, true, testN), 0.2802507761872869 );
        delete testN;
    }

    TEST(NetMath, lecuntanh_filter) {
        Filter* testF = new Filter();
        EXPECT_EQ( NetMath::lecuntanh(2.0, false, testF), 1.4929388053842507 );
        EXPECT_EQ( NetMath::lecuntanh(-2.0, false, testF), -1.4929388053842507 );
        EXPECT_EQ( NetMath::lecuntanh(2.0, true, testF), 0.2802507761872869 );
        EXPECT_EQ( NetMath::lecuntanh(-2.0, true, testF), 0.2802507761872869 );
        delete testF;
    }

    TEST(NetMath, relu) {
        Neuron* testN = new Neuron();
        EXPECT_EQ( NetMath::relu(2, false, testN), 2 );
        EXPECT_EQ( NetMath::relu(-2, false, testN), 0 );
        EXPECT_EQ( NetMath::relu(2, true, testN), 1 );
        EXPECT_EQ( NetMath::relu(-2, true, testN), 0 );
        delete testN;
    }

    TEST(NetMath, relu_filter) {
        Filter* testF = new Filter();
        EXPECT_EQ( NetMath::relu(2, false, testF), 2 );
        EXPECT_EQ( NetMath::relu(-2, false, testF), 0 );
        EXPECT_EQ( NetMath::relu(2, true, testF), 1 );
        EXPECT_EQ( NetMath::relu(-2, true, testF), 0 );
        delete testF;
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

    TEST(NetMath, lrelu_filter) {
        Filter* testF = new Filter();
        testF->lreluSlope = -0.0005;
        EXPECT_EQ( NetMath::lrelu(2, false, testF), 2 );
        EXPECT_EQ( NetMath::lrelu(-2, false, testF), -0.001 );
        EXPECT_EQ( NetMath::lrelu(2, true, testF), 1 );
        EXPECT_EQ( NetMath::lrelu(-2, true, testF), -0.0005 );
        delete testF;
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

    TEST(NetMath, rrelu_filter) {
        Filter* testF = new Filter();
        testF->rreluSlope = 0.0005;
        EXPECT_EQ( NetMath::rrelu(2, false, testF), 2 );
        EXPECT_EQ( NetMath::rrelu(-2, false, testF), 0.0005 );
        EXPECT_EQ( NetMath::rrelu(2, true, testF), 1 );
        EXPECT_EQ( NetMath::rrelu(-2, true, testF), 0.0005 );
        delete testF;
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

    TEST(NetMath, elu_filter) {
        Filter* testF = new Filter();
        testF->eluAlpha = 1;
        EXPECT_EQ( NetMath::elu(2, false, testF), 2 );
        EXPECT_EQ( NetMath::elu(-0.25, false, testF), -0.22119921692859512 );
        EXPECT_EQ( NetMath::elu(2, true, testF), 1 );
        EXPECT_EQ( NetMath::elu(-0.5, true, testF), 0.6065306597126334 );
        delete testF;
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
            testN->init(0, 5);

            testF = new Filter();
            testF->weights = { {{1,1,1},{1,1,1},{1,1,1}} };
            testF->init(0);
            testF->bias = 0.1;
        }

        virtual void TearDown() {
            delete testN;
            delete testF;
            Network::deleteNetwork();
        }

        Network* net;
        Neuron* testN;

        Filter* testF;
    };

    // Doubles a value when the gain is 2 and learningRate 1
    TEST_F(GainFixture, gain_1) {
        testN->biasGain = 2;
        EXPECT_EQ( NetMath::gain(0, (double)10, (double)5, testN, -1), 20 );
        testF->biasGain = 2;
        EXPECT_EQ( NetMath::gain(0, (double)10, (double)5, testF, -1, -1, -1), 20 );
    }

    // Halves a value when the gain is -5 and learningRate 0.1
    TEST_F(GainFixture, gain_2) {
        net->learningRate = 0.1;
        testN->biasGain = -5;
        EXPECT_NEAR( NetMath::gain(0, (double)5, (double)5, testN, -1), 2.5, 1e-6 );
        testF->biasGain = -5;
        EXPECT_NEAR( NetMath::gain(0, (double)5, (double)5, testF, -1, -1, -1), 2.5, 1e-6 );
    }

    // Increments a neuron's biasGain by 0.05 when the bias value doesn't change sign
    TEST_F(GainFixture, gain_3) {
        testN->biasGain = 1;
        NetMath::gain(0, (double)0.1, (double)1, testN, -1);
        EXPECT_EQ( testN->biasGain, 1.05 );

        testF->biasGain = 1;
        NetMath::gain(0, (double)0.1, (double)1, testF, -1, -1, -1);
        EXPECT_EQ( testF->biasGain, 1.05 );
    }

    // Does not increase the gain to more than 5
    TEST_F(GainFixture, gain_4) {
        testN->biasGain = 4.99;
        NetMath::gain(0, (double)0.1, (double)1, testN, -1);
        EXPECT_EQ( testN->biasGain, 5 );

        testF->biasGain = 4.99;
        NetMath::gain(0, (double)0.1, (double)1, testF, -1, -1, -1);
        EXPECT_EQ( testF->biasGain, 5 );
    }

    // Multiplies a neuron's bias gain by 0.95 when the value changes sign
    TEST_F(GainFixture, gain_5) {
        net->learningRate = -10;
        testN->biasGain = 1;
        NetMath::gain(0, (double)0.1, (double)1, testN, -1);
        EXPECT_EQ( testN->biasGain, 0.95 );

        testF->biasGain = 1;
        NetMath::gain(0, (double)0.1, (double)1, testF, -1, -1, -1);
        EXPECT_EQ( testF->biasGain, 0.95 );
    }

    // Does not reduce the bias gain to less than 0.5
    TEST_F(GainFixture, gain_6) {
        net->learningRate = -10;
        testN->biasGain = 0.51;
        NetMath::gain(0, (double)0.1, (double)1, testN, -1);
        EXPECT_EQ( testN->biasGain, 0.5 );

        testF->biasGain = 0.51;
        NetMath::gain(0, (double)0.1, (double)1, testF, -1, -1, -1);
        EXPECT_EQ( testF->biasGain, 0.5 );
    }

    // Increases weight gain the same way as the bias gain
    TEST_F(GainFixture, gain_7) {
        testN->weightGain = {1, 4.99};
        NetMath::gain(0, (double)0.1, (double)1, testN, 0);
        NetMath::gain(0, (double)0.1, (double)1, testN, 1);
        EXPECT_EQ( testN->weightGain[0], 1.05 );
        EXPECT_EQ( testN->weightGain[1], 5 );

        testF->weights = { {{0.1,0.1,  1},{1,1,1},{1,1,1}} };
        testF->weightGain = { {{1,4.99  ,1},{1,1,1},{1,1,1}} };
        NetMath::gain(0, (double)0.1, (double)1, testF, 0, 0, 0);
        NetMath::gain(0, (double)0.1, (double)1, testF, 0, 0, 1);
        EXPECT_EQ( testF->weightGain[0][0][0], 1.05 );
        EXPECT_EQ( testF->weightGain[0][0][1], 5 );
    }

    // Decreases weight gain the same way as the bias gain
    TEST_F(GainFixture, gain_8) {
        net->learningRate = -10;
        testN->weightGain = {1, 0.51};
        NetMath::gain(0, (double)0.1, (double)1, testN, 0);
        NetMath::gain(0, (double)0.1, (double)1, testN, 1);
        EXPECT_EQ( testN->weightGain[0], 0.95 );
        EXPECT_EQ( testN->weightGain[1], 0.5 );

        testF->weights = { {{0.1,0.1,  1},{1,1,1},{1,1,1}} };
        testF->weightGain = { {{1,0.51  ,1},{1,1,1},{1,1,1}} };
        NetMath::gain(0, (double)0.1, (double)1, testF, 0, 0, 0);
        NetMath::gain(0, (double)0.1, (double)1, testF, 0, 0, 1);
        EXPECT_EQ( testF->weightGain[0][0][0], 0.95 );
        EXPECT_EQ( testF->weightGain[0][0][1], 0.5 );
    }

    class AdagradFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->learningRate = 2;
            testN = new Neuron();
            testN->init(0, 5);
            testN->biasCache = 0;

            testF = new Filter();
            testF->weights = { {{1,1,1},{1,1,1},{1,1,1}} };
            testF->init(0);
            testF->biasCache = 0;
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete testN;
            delete testF;
        }

        Network* net;
        Neuron* testN;
        Filter* testF;
    };

    // Increments the neuron's biasCache by the square of its deltaBias
    TEST_F(AdagradFixture, adagrad_1) {
        NetMath::adagrad(0, (double)1, (double)3, testN, -1);
        EXPECT_EQ( testN->biasCache, 9 );
        NetMath::adagrad(0, (double)1, (double)3, testF, -1, -1, -1);
        EXPECT_EQ( testF->biasCache, 9 );
    }

    // Returns a new value matching the formula for adagrad
    TEST_F(AdagradFixture, adagrad_2) {
        net->learningRate = 0.5;
        EXPECT_NEAR( NetMath::adagrad(0, (double)1, (double)3, testN, -1), 1.5, 1e-3 );
        EXPECT_NEAR( NetMath::adagrad(0, (double)1, (double)3, testF, -1, -1, -1), 1.5, 1e-3 );
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

        testF->weightsCache = { {{0,1,2},{1,1,1},{1,1,1}} };
        double result4 = NetMath::adagrad(0, (double)1, (double)3, testF, 0, 0, 0);
        double result5 = NetMath::adagrad(0, (double)1, (double)4, testF, 0, 0, 1);
        double result6 = NetMath::adagrad(0, (double)1, (double)2, testF, 0, 0, 2);
        EXPECT_EQ( testF->weightsCache[0][0][0], 9 );
        EXPECT_EQ( testF->weightsCache[0][0][1], 17 );
        EXPECT_EQ( testF->weightsCache[0][0][2], 6 );
        EXPECT_NEAR( result4, 3.0, 1e-2 );
        EXPECT_NEAR( result5, 2.9, 1e-1 );
        EXPECT_NEAR( result6, 2.6, 1e-1 );
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
            testN->init(0, 5);
            testN->biasCache = 10;

            testF = new Filter();
            testF->weights = { {{1,1,1},{1,1,1},{1,1,1}} };
            testF->init(0);
            testF->biasCache = 10;
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete testN;
            delete testF;
        }

        Network* net;
        Neuron* testN;
        Filter* testF;
    };

    // Sets the cache value to the correct value, following the rmsprop formula
    TEST_F(RMSPropFixture, rmsprop_1) {
        net->learningRate = 2;
        NetMath::rmsprop(0, (double)1, (double)3, testN, -1);
        EXPECT_NEAR(testN->biasCache, 9.99, 1e-3);

        NetMath::rmsprop(0, (double)1, (double)3, testF, -1, -1, -1);
        EXPECT_NEAR(testF->biasCache, 9.99, 1e-3);
    }

    // Returns a new value matching the formula for rmsprop, using this new cache value
    TEST_F(RMSPropFixture, rmsprop_2) {
        EXPECT_NEAR( NetMath::rmsprop(0, (double)1, (double)3, testN, -1), 1.47, 1e-2);
        EXPECT_NEAR( NetMath::rmsprop(0, (double)1, (double)3, testF, -1, -1, -1), 1.47, 1e-2);
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

        testF->weightsCache = { {{0,1,2},{1,1,1},{1,1,1}} };
        double result4 = NetMath::rmsprop(0, (double)1, (double)3, testF, 0, 0, 0);
        double result5 = NetMath::rmsprop(0, (double)1, (double)4, testF, 0, 0, 1);
        double result6 = NetMath::rmsprop(0, (double)1, (double)2, testF, 0, 0, 2);
        EXPECT_NEAR( testF->weightsCache[0][0][0], 0.09, 1e-2 );
        EXPECT_NEAR( testF->weightsCache[0][0][1], 1.15, 1e-2 );
        EXPECT_NEAR( testF->weightsCache[0][0][2], 2.02, 1e-2 );
        EXPECT_NEAR( result4, 6.0, 1e-2 );
        EXPECT_NEAR( result5, 2.9, 0.1 );
        EXPECT_NEAR( result6, 1.7, 0.1 );
    }

    class AdamFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->learningRate = 0.01;
            testN = new Neuron();
            testN->init(0, 5);

            testF = new Filter();
            testF->weights = { {{1,1,1},{1,1,1},{1,1,1}} };
            testF->init(0);
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete testN;
            delete testF;
        }

        Network* net;
        Neuron* testN;
        Filter* testF;
    };

    // It sets the neuron.m to the correct value, following the formula
    TEST_F(AdamFixture, adam_1) {
        testN->m = 0.1;
        NetMath::adam(0, (double)1, (double)0.2, testN, -1);
        EXPECT_DOUBLE_EQ( testN->m, 0.11 );

        testF->m = 0.1;
        NetMath::adam(0, (double)1, (double)0.2, testF, -1, -1, -1);
        EXPECT_DOUBLE_EQ( testF->m, 0.11 );
    }

    // It sets the neuron.v to the correct value, following the formula
    TEST_F(AdamFixture, adam_2) {
        testN->v = 0.1;
        NetMath::adam(0, (double)1, (double)0.2, testN, -1);
        EXPECT_NEAR( testN->v, 0.09994, 1e-3 );

        testF->v = 0.1;
        NetMath::adam(0, (double)1, (double)0.2, testF, -1, -1, -1);
        EXPECT_NEAR( testF->v, 0.09994, 1e-3 );
    }

    // Calculates a value correctly, following the formula
    TEST_F(AdamFixture, adam_3) {
        net->iterations = 2;
        testN->m = 0.121;
        testN->v = 0.045;
        EXPECT_NEAR( NetMath::adam(0, (double)-0.3, (double)0.02, testN, -1), -0.298943, 1e-5 );

        testF->m = 0.121;
        testF->v = 0.045;
        EXPECT_NEAR( NetMath::adam(0, (double)-0.3, (double)0.02, testF, -1, -1, -1), -0.298943, 1e-5 );
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
            testN->init(0, 5);
            testN->biasCache = 0.5;

            testF = new Filter();
            testF->weights = { {{1,1},{1,1}} };
            testF->adadeltaCache = { {{1,1},{1,1}} };
            testF->init(0);
            testF->biasCache = 0.5;
        }

        virtual void TearDown() {
            Network::deleteNetwork();
            delete testN;
            delete testF;
        }

        Network* net;
        Neuron* testN;
        Filter* testF;
    };

    // Sets the neuron.biasCache to the correct value, following the adadelta formula
    TEST_F(AdadeltaFixture, adadelta_1) {
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, -1);
        EXPECT_NEAR( testN->biasCache, 0.477, 1e-3 );

        NetMath::adadelta(0, (double)0.5, (double)0.2, testF, -1, -1, -1);
        EXPECT_NEAR( testF->biasCache, 0.477, 1e-3 );
    }

    // Sets the weightsCache to the correct value, following the adadelta formula, same as biasCache
    TEST_F(AdadeltaFixture, adadelta_2) {
        testN->weightsCache = {0.5, 0.75};
        testN->adadeltaCache = {0, 0};
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 0);
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 1);
        EXPECT_NEAR( testN->weightsCache[0], 0.477, 1e-3 );
        EXPECT_NEAR( testN->weightsCache[1], 0.7145, 1e-4 );

        testF->weightsCache = { {{0.5,0.75},{1,1}} };
        testF->adadeltaCache = { {{0,0},{1,1}} };
        NetMath::adadelta(0, (double)0.5, (double)0.2, testF, 0, 0, 0);
        NetMath::adadelta(0, (double)0.5, (double)0.2, testF, 0, 0, 1);
        EXPECT_NEAR( testF->weightsCache[0][0][0], 0.477, 1e-3 );
        EXPECT_NEAR( testF->weightsCache[0][0][1], 0.7145, 1e-4 );
    }

    // Creates a value for the bias correctly, following the formula
    TEST_F(AdadeltaFixture, adadelta_3) {
        testN->adadeltaBiasCache = 0.25;
        EXPECT_NEAR( NetMath::adadelta(0, (double)0.5, (double)0.2, testN, -1), 0.64479, 1e-5 );

        testF->adadeltaBiasCache = 0.25;
        EXPECT_NEAR( NetMath::adadelta(0, (double)0.5, (double)0.2, testF, -1, -1, -1), 0.64479, 1e-5 );
    }

    // Creates a value for the weight correctly, the same was as the bias
    TEST_F(AdadeltaFixture, adadelta_4) {
        testN->weightsCache = {0.5, 0.75};
        testN->adadeltaCache = {0.1, 0.2};
        EXPECT_NEAR( NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 0), 0.59157, 1e-3 );
        EXPECT_NEAR( NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 1), 0.60581, 1e-3 );

        testF->weightsCache = { {{0.5,0.75},{1,1}} };
        testF->adadeltaCache = { {{0.1,0.2},{1,1}} };
        EXPECT_NEAR( NetMath::adadelta(0, (double)0.5, (double)0.2, testF, 0, 0, 0), 0.59157, 1e-3 );
        EXPECT_NEAR( NetMath::adadelta(0, (double)0.5, (double)0.2, testF, 0, 0, 1), 0.60581, 1e-3 );
    }

    // Updates the neuron.adadeltaBiasCache with the correct value, following the formula
    TEST_F(AdadeltaFixture, adadelta_5) {
        testN->adadeltaBiasCache = 0.25;
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, -1);
        EXPECT_NEAR( testN->adadeltaBiasCache, 0.2395, 1e-2 );

        testF->adadeltaBiasCache = 0.25;
        NetMath::adadelta(0, (double)0.5, (double)0.2, testF, -1, -1, -1);
        EXPECT_NEAR( testF->adadeltaBiasCache, 0.2395, 1e-2 );
    }

    // Updates the neuron.adadeltaCache with the correct value, following the formula, same as adadeltaBiasCache
    TEST_F(AdadeltaFixture, adadelta_6) {
        testN->weightsCache = {0.5, 0.75};
        testN->adadeltaCache = {0.1, 0.2};
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 0);
        NetMath::adadelta(0, (double)0.5, (double)0.2, testN, 1);
        EXPECT_NEAR( testN->adadeltaCache[0], 0.097, 0.1 );
        EXPECT_NEAR( testN->adadeltaCache[1], 0.192, 0.1 );

        testF->weightsCache = { {{0.5,0.75},{1,1}} };
        testF->adadeltaCache = { {{0.1,0.2},{1,1}} };
        NetMath::adadelta(0, (double)0.5, (double)0.2, testF, 0, 0, 0);
        NetMath::adadelta(0, (double)0.5, (double)0.2, testF, 0, 0, 1);
        EXPECT_NEAR( testF->adadeltaCache[0][0][0], 0.097, 0.1 );
        EXPECT_NEAR( testF->adadeltaCache[0][0][1], 0.192, 0.1 );
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
            l2->weights = {{2}};
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
        EXPECT_EQ( l2->weights[0][0], 0.7071067811865475 );
    }

    // Does not scale weights if their L2 doesn't exceed the configured max norm threshold
    TEST_F(MaxNormFixture, maxNorm_3) {
        net->maxNorm = 1000;
        NetMath::maxNorm(0);
        EXPECT_EQ( l2->weights[0][0], 2 );
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

        std::vector<double> values = NetMath::uniform(0, 0, 100);

        for (int i=0; i<100; i++) {
            EXPECT_LE( values[i], 0.1 );
            EXPECT_GT( values[i], -0.1 );
        }
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

        for (int i=0; i<1000; i++) {
            EXPECT_LE( values[i], 0.5 );
            EXPECT_GT( values[i], -0.5 );
        }
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

        for (int i=0; i<1000; i++) {
            EXPECT_LE( values[i], 0.5 );
            EXPECT_GT( values[i], -0.5 );
        }
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


    TEST(NetMath, maxPool) {

        Network::deleteNetwork();
        Network::newNetwork();
        Network* net = Network::getInstance(0);
        net->updateFnIndex = 0;
        net->activation = &NetMath::sigmoid;
        net->weightInitFn = &NetMath::uniform;

        std::vector<double> testData = {
            1,2,4,5,7,8,10,11,13,14,16,17,
            1,2,4,5,7,8,10,11,13,14,16,17,

            1,2,4,5,7,9,10,11,13,14,16,17,
            2,3,5,6,8,8,11,12,14,15,17,18,

            1,2,4,5,7,8,10,11,13,14,16,17,
            2,3,5,6,8,9,11,12,14,15,17,18,

            1,2,4,5,7,8,10,11,13,14,16,17,
            1,2,4,5,7,8,10,11,13,14,16,17,

            1,2,4,5,7,9,10,11,13,14,16,17,
            2,3,5,6,8,8,11,12,14,15,17,18,

            1,2,4,5,7,8,10,11,13,14,16,17,
            2,3,5,6,8,9,11,12,14,15,17,18
        };

        std::vector<std::vector<std::vector<double> > > expectedActivationMap = {{
            {2,5,8,11,14,17},
            {3,6,9,12,15,18},
            {3,6,9,12,15,18},
            {2,5,8,11,14,17},
            {3,6,9,12,15,18},
            {3,6,9,12,15,18}
        }};
        std::vector<std::vector<std::vector<std::vector<int> > > > expectedIndeces = {{
            {{0,1},{0,1},{0,1},{0,1},{0,1},{0,1}},
            {{1,1},{1,1},{0,1},{1,1},{1,1},{1,1}},
            {{1,1},{1,1},{1,1},{1,1},{1,1},{1,1}},
            {{0,1},{0,1},{0,1},{0,1},{0,1},{0,1}},
            {{1,1},{1,1},{0,1},{1,1},{1,1},{1,1}},
            {{1,1},{1,1},{1,1},{1,1},{1,1},{1,1}}
        }};

        PoolLayer* layer = new PoolLayer(0, 2);
        layer->channels = 1;
        layer->stride = 2;
        layer->inMapValuesCount = 144;
        layer->prevLayerOutWidth = 12;
        layer->outMapSize = 6;

        FCLayer* prevLayer = new FCLayer(0, 144);
        prevLayer->assignNext(layer);
        layer->assignPrev(prevLayer);

        prevLayer->init(0);
        layer->init(1);

        prevLayer->actvns = {};
        for (int i=0; i<144; i++) {
            prevLayer->actvns[i] = testData[i];
        }

        NetMath::maxPool(layer, 0);

        EXPECT_EQ( layer->activations, expectedActivationMap );
        EXPECT_EQ( layer->indeces, expectedIndeces );

        delete layer;
        delete prevLayer;
    }


    TEST(NetMath, softmax_1) {
        std::vector<double> values = {1, 2, 3, 4, 1, 2, 3};
        std::vector<double> expected = {0.02364054302159139, 0.06426165851049616, 0.17468129859572226, 0.47483299974438037, 0.02364054302159139, 0.06426165851049616, 0.17468129859572226};
        EXPECT_EQ( NetMath::softmax(values), expected );
    }

    TEST(NetMath, softmax_2) {
        std::vector<double> values = {23, 54, 167, 3};
        std::vector<double> expected = {2.8946403116483003e-63, 8.408597124803643e-50, 1, 5.96629836401057e-72};
        EXPECT_EQ( NetMath::softmax(values), expected );
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

            bool bad = values[i] != original[0] && values[i] != original[1] && values[i] != original[2];

            EXPECT_FALSE( bad );
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

    // Returns a map with 2 levels of 0 values padded, when zero padding of 2 is given
    TEST_F(AddZPaddingFixture, addZeroPadding_3) {
        std::vector<std::vector<double> > res = NetUtil::addZeroPadding(testData, 2);
        std::vector<double> zeroRow = {0,0,0,0,0,0,0,0,0};

        EXPECT_EQ( res.size(), 9 );
        EXPECT_EQ( res[0], zeroRow );
        EXPECT_EQ( res[1], zeroRow );
        EXPECT_NE( res[2], zeroRow );

        for (int r=0; r<9; r++) {
            EXPECT_EQ( res[r].size(), 9 );
        }

        EXPECT_NE( res[res.size()-3], zeroRow );
        EXPECT_EQ( res[res.size()-2], zeroRow );
        EXPECT_EQ( res[res.size()-1], zeroRow );
    }

    // Returns a map with 3 levels of 0 values padded, when zero padding of 3 is given
    TEST_F(AddZPaddingFixture, addZeroPadding_4) {
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
    TEST_F(AddZPaddingFixture, addZeroPadding_5) {
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

    // Calculates values correctly (Example a)
    TEST(NetUtil, convolve_1) {
        std::vector<double> testInputa = {0,0,2,2,2, 1,1,0,2,0, 1,2,1,1,2, 0,1,2,2,1, 1,2,0,0,1};
        std::vector<std::vector<std::vector<double> > > testWeightsa = {{{-1,0,-1},{1,0,1},{1,-1,0}}};
        std::vector<std::vector<double> > expecteda = {{0,4,5}, {2,0,1}, {2,0,-1}};
        std::vector<std::vector<double> > res = NetUtil::convolve(testInputa, 1, testWeightsa, 1, 2, 1);
        EXPECT_EQ( res, expecteda );
    }

    // Calculates values correctly (Example b)
    TEST(NetUtil, convolve_2) {
        std::vector<double> testInputb = {2,2,1,1,2, 1,1,2,0,0, 2,0,0,2,2, 1,2,2,1,1, 1,1,2,0,1};
        std::vector<std::vector<std::vector<double> > > testWeightsb = {{{0,1,1},{1,-1,-1},{-1,1,0}}};
        std::vector<std::vector<double> > expectedb = {{-2,2,0},{2,1,1},{2,3,1}};
        std::vector<std::vector<double> > res = NetUtil::convolve(testInputb, 1, testWeightsb, 1, 2, 1);
        EXPECT_EQ( res, expectedb );
    }

    // Calculates values correctly (Example c)
    TEST(NetUtil, convolve_3) {
        std::vector<double> testInputc = {0,1,1,0,0, 1,2,0,2,0, 2,0,1,2,0, 2,0,1,0,1, 0,1,2,2,1};
        std::vector<std::vector<std::vector<double> > > testWeightsc = {{{-1,0,-1},{1,0,0},{1,0,0}}};
        std::vector<std::vector<double> > expectedc = {{1,4,3},{-1,-3,1},{1,2,3}};
        std::vector<std::vector<double> > res = NetUtil::convolve(testInputc, 1, testWeightsc, 1, 2, 1);
        EXPECT_EQ( res, expectedc );
    }

    // Calculates values correctly (Example 1)
    TEST(NetUtil, convolve_4) {
        std::vector<double> testInput = {0,0,2,2,2, 1,1,0,2,0, 1,2,1,1,2, 0,1,2,2,1, 1,2,0,0,1, 2,2,1,1,2, 1,1,2,0,0, 2,0,0,2,2, 1,2,2,1,1, 1,1,2,0,1, 0,1,1,0,0, 1,2,0,2,0, 2,0,1,2,0, 2,0,1,0,1, 0,1,2,2,1};
        std::vector<std::vector<std::vector<double> > > testWeights1 = {{{-1,0,-1},{1,0,1},{1,-1,0}},   {{0,1,1},{1,-1,-1},{-1,1,0}},  {{-1,0,-1},{1,0,0},{1,0,0}}};
        std::vector<std::vector<double> > expected1 = {{-3,8,6},{1,-4,1},{3,3,1}};
        std::vector<std::vector<double> > res = NetUtil::convolve(testInput, 1, testWeights1, 3, 2, 1);
        EXPECT_EQ( res, expected1 );
    }

    // Calculates values correctly (Example 2)
    TEST(NetUtil, convolve_5) {
        std::vector<double> testInput = {0,0,2,2,2, 1,1,0,2,0, 1,2,1,1,2, 0,1,2,2,1, 1,2,0,0,1, 2,2,1,1,2, 1,1,2,0,0, 2,0,0,2,2, 1,2,2,1,1, 1,1,2,0,1, 0,1,1,0,0, 1,2,0,2,0, 2,0,1,2,0, 2,0,1,0,1, 0,1,2,2,1};
        std::vector<std::vector<std::vector<double> > > testWeights2 = {{{-1,0,1},{-1,1,1},{1,0,0}},   {{0,-1,1},{1,-1,1},{-1,1,-1}},  {{0,0,0},{0,-1,-1},{0,0,1}}};
        std::vector<std::vector<double> > expected2 = {{1,9,1},{-1,-2,1},{4,-7,-4}};
        std::vector<std::vector<double> > res = NetUtil::convolve(testInput, 1, testWeights2, 3, 2, 0);
        EXPECT_EQ( res, expected2 );
    }

    TEST(NetUtil, createVolume) {
        std::vector<std::vector<std::vector<double> > > expected1 = {{{1,1,1},{1,1,1},{1,1,1}}, {{1,1,1},{1,1,1},{1,1,1}}};
        std::vector<std::vector<std::vector<double> > > expected2 = {{{0,0},{0,0},{0,0}}};

        EXPECT_EQ( NetUtil::createVolume<double>(2, 3, 3, 1), expected1 );
        EXPECT_EQ( NetUtil::createVolume<double>(1, 3, 2, 0), expected2 );
    }


    class BuildConvDWeightsFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            Network* net = Network::getInstance(0);
            net->weightInitFn = &NetMath::uniform;
            net->channels = 3;
            net->miniBatchSize = 1;

            layer = new ConvLayer(0, 2);
            layer->channels = 3;
            layer->filterSize = 3;
            layer->zeroPadding = 1;
            layer->stride = 2;
            layer->inMapValuesCount = 25;
            layer->outMapSize = 14;
            prevLayer = new FCLayer(0, 75);

            prevLayer->init(0);
            layer->assignPrev(prevLayer);
            layer->init(1);

            prevLayer->actvns = {};
            for (int n=0; n<prevLayer->neurons.size(); n++) {
                prevLayer->actvns[n] = n+1;
            }

            layer->filters[0]->errorMap = {{0.1, 0.6, 0.2}, {0.7, 0.3, 0.8}, {0.4, 0.9, 0.5}};
            layer->filters[1]->errorMap = {{-0.5, 0, -0.4}, {0.1, -0.3, 0.2}, {-0.2, 0.3, -0.1}};

            for (int c=0; c<layer->filters[0]->deltaWeights.size(); c++) {
                for (int r=0; r<layer->filters[0]->deltaWeights[0].size(); r++) {
                    for (int v=0; v<layer->filters[0]->deltaWeights[0].size(); v++) {
                        layer->filters[0]->deltaWeights[c][r][v] = 0;
                        layer->filters[1]->deltaWeights[c][r][v] = 0;
                    }
                }
            }

        }

        virtual void TearDown() {
            Network::deleteNetwork();
        }

        ConvLayer* layer;
        FCLayer* prevLayer;
    };

    // Sets the filter 1 deltaBias to 4.5 and filter 2 deltaBias to -0.9
    TEST_F(BuildConvDWeightsFixture, buildConvDWeights_1) {
        NetUtil::buildConvDWeights(layer);
        EXPECT_EQ( layer->filters[0]->deltaBias, 4.5 );
        EXPECT_EQ( layer->filters[1]->deltaBias, -0.9 );
    }

    // Sets the filter 1 deltaWeights to hand worked out values
    TEST_F(BuildConvDWeightsFixture, buildConvDWeights_2) {
        std::vector<std::vector<std::vector<double> > > expected = {
            {{34.1, 47.2, 31.5}, {48.6, 68.1, 45.6}, {26.3, 40, 23.7}},
            {{96.6, 137.2, 89}, {131.1, 180.6, 120.6}, {73.8, 107.5, 66.2}},
            {{159.1, 227.2, 146.5}, {213.6, 293.1, 195.6}, {121.3, 175, 108.7}}
        };
        NetUtil::buildConvDWeights(layer);

        for (int c=0; c<layer->filters[0]->deltaWeights.size(); c++) {
            for (int r=0; r<layer->filters[0]->deltaWeights[0].size(); r++) {
                for (int v=0; v<layer->filters[0]->deltaWeights[0].size(); v++) {
                    EXPECT_NEAR( layer->filters[0]->deltaWeights[c][r][v], expected[c][r][v], 1e-8 );
                }
            }
        }
    }

    // Sets the filter 2 deltaWeights to hand worked out values
    TEST_F(BuildConvDWeightsFixture, buildConvDWeights_3) {
        std::vector<std::vector<std::vector<double> > > expected = {
            {{2.9, 0.4, 0.3}, {1.8, -2.1, -1.2}, {-4.9, -6.8, -7.5}},
            {{5.4, 0.4, -2.2}, {-5.7, -24.6, -16.2}, {-17.4, -29.3, -25}},
            {{7.9, 0.4, -4.7}, {-13.2, -47.1, -31.2}, {-29.9, -51.8, -42.5}}
        };
        NetUtil::buildConvDWeights(layer);

        for (int c=0; c<layer->filters[1]->deltaWeights.size(); c++) {
            for (int r=0; r<layer->filters[1]->deltaWeights[0].size(); r++) {
                for (int v=0; v<layer->filters[1]->deltaWeights[0].size(); v++) {
                    EXPECT_NEAR( layer->filters[1]->deltaWeights[c][r][v], expected[c][r][v], 1e-8 );
                }
            }
        }
    }


    class BuildConvErrorMapFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            net = Network::getInstance(0);
            net->weightInitFn = &NetMath::uniform;

            layer = new ConvLayer(0, 1);
            layer->filters.push_back(new Filter());
            layer->filters[0]->errorMap = {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}};

            nextLayerA = new ConvLayer(0, 1);
            nextLayerA->filterSize = 3;
            nextLayerA->zeroPadding = 1;
            nextLayerA->stride = 2;

            nextLayerB = new ConvLayer(0, 2);
            nextLayerB->filterSize = 3;
            nextLayerB->zeroPadding = 1;
            nextLayerB->stride = 2;

            nextLayerC = new ConvLayer(0, 1);
            nextLayerC->filterSize = 3;
            nextLayerC->zeroPadding = 1;
            nextLayerC->stride = 1;

            nlFilterA = new Filter();
            nlFilterA->errorMap = {{0.5, -0.2, 0.1}, {0, -0.4, -0.1}, {0.2, 0.6, 0.3}};
            nlFilterA->weights = {{{-1, 0, -1}, {1, 0, 1}, {1, -1, 0}}};

            nlFilterB = new Filter();
            nlFilterB->errorMap = {{0.1, 0.4, 0.2}, {-0.1,0.2,-0.3}, {0, -0.4, 0.5}};
            nlFilterB->weights = {{{1, 1, 0}, {-1, 1, 0}, {1, -1, 1}}};

            nlFilterC = new Filter();
            nlFilterC->errorMap = {{0.1,0.4,-0.2,0.3,0},{0.9,0.2,-0.7,1.1,0.6},{0.4,0,0.3,-0.8,0.1},{0.2,0.3,0.1,-0.1,0.5},{-0.3,0.4,0.5,-0.2,0.3}};
            nlFilterC->weights = {{{1, 1, 0}, {-1, 1, 0}, {1, -1, 1}}};
        }

        virtual void TearDown() {
            delete layer;
            delete nextLayerA;
            delete nextLayerB;
            delete nextLayerC;
        }

        Network* net;
        ConvLayer* layer;
        ConvLayer* nextLayerA;
        ConvLayer* nextLayerB;
        ConvLayer* nextLayerC;
        Filter* nlFilterA;
        Filter* nlFilterB;
        Filter* nlFilterC;
    };

    // Calculates an error map correctly, using just one channel from 1 filter in next layer (Example 1)
    TEST_F(BuildConvErrorMapFixture, buildConvErrorMap_1) {

        nextLayerA->filters = {nlFilterA};
        layer->assignNext(nextLayerA);

        std::vector<std::vector<double> > expectedA = {{0,0.3,0,-0.1,0},{-0.5,0.2,0.2,0.6,-0.1},{0,-0.4,0,-0.5,0},{0,-1.2,0.4,-1,0.1},{0,0.8,0,0.9,0}};
        layer->filters[0]->errorMap = NetUtil::buildConvErrorMap(5+2, nextLayerA, 0);

        EXPECT_EQ( layer->filters[0]->errorMap.size(), 5 );
        EXPECT_EQ( layer->filters[0]->errorMap[0].size(), 5 );

        for (int r=0; r<5; r++) {
            for (int c=0; c<5; c++) {
                EXPECT_NEAR( layer->filters[0]->errorMap[r][c], expectedA[r][c], 1e-8 );
            }
        }
    }

    // Clears the filter errorMap values first (by getting the same result with different initial errorMap values, using Example 1)
    TEST_F(BuildConvErrorMapFixture, buildConvErrorMap_2) {
        layer->filters[0]->errorMap = {{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}};
        nextLayerA->filters = {nlFilterA};
        layer->assignNext(nextLayerA);
        std::vector<std::vector<double> > expectedA = {{0,0.3,0,-0.1,0},{-0.5,0.2,0.2,0.6,-0.1},{0,-0.4,0,-0.5,0},{0,-1.2,0.4,-1,0.1},{0,0.8,0,0.9,0}};
        layer->filters[0]->errorMap = NetUtil::buildConvErrorMap(5+2, nextLayerA, 0);

        for (int r=0; r<5; r++) {
            for (int c=0; c<5; c++) {
                EXPECT_NEAR( layer->filters[0]->errorMap[r][c], expectedA[r][c], 1e-8 );
            }
        }
    }

    // Calculates an error map correctly, using just one channel from 1 filter in next layer (Example 2)
    TEST_F(BuildConvErrorMapFixture, buildConvErrorMap_3) {
        nextLayerA->filters = {nlFilterB};
        layer->assignNext(nextLayerA);
        std::vector<std::vector<double> > expectedB = {{0.1,-0.4,0.4,-0.2,0.2},{-0.2,0.7,-0.2,0.3,-0.5},{-0.1,-0.2,0.2,0.3,-0.3},{0.1,-0.3,-0.6,0.4,0.8},{0,0.4,-0.4,-0.5,0.5}};
        layer->filters[0]->errorMap = NetUtil::buildConvErrorMap(5+2, nextLayerA, 0);

        for (int r=0; r<5; r++) {
            for (int c=0; c<5; c++) {
                EXPECT_NEAR( layer->filters[0]->errorMap[r][c], expectedB[r][c], 1e-8 );
            }
        }
    }

    // Calculates an error map correctly, using two channels, from 2 filters in the next layer
    TEST_F(BuildConvErrorMapFixture, buildConvErrorMap_4) {
        nextLayerB->filters = {nlFilterA, nlFilterB};
        layer->assignNext(nextLayerB);
        std::vector<std::vector<double> > expectedC = {{0.1,-0.1,0.4,-0.3,0.2},{-0.7,0.9,0,0.9,-0.6},{-0.1,-0.6,0.2,-0.2,-0.3},{0.1,-1.5,-0.2,-0.6,0.9},{0,1.2,-0.4,0.4,0.5}};
        layer->filters[0]->errorMap = NetUtil::buildConvErrorMap(5+2, nextLayerB, 0);

        for (int r=0; r<5; r++) {
            for (int c=0; c<5; c++) {
                EXPECT_NEAR( layer->filters[0]->errorMap[r][c], expectedC[r][c], 1e-8 );
            }
        }
    }

    // Calculates an error map correctly, using 1 channel where the stride is 1, not 2
    TEST_F(BuildConvErrorMapFixture, buildConvErrorMap_5) {
        nextLayerC->filters = {nlFilterC};
        layer->assignNext(nextLayerC);
        std::vector<std::vector<double> > expectedD = {{0.8,0.1,-0.1,2.0,0.6},{1.4,0.7,-1.4,-0.7,1},{0.2,0.1,3.1,-1.7,1.1},{-0.4,1.8,-0.6,0.7,-0.1},{-0.6,-0.1,0.8,0.2,-0.3}};
        layer->filters[0]->errorMap = NetUtil::buildConvErrorMap(5+2, nextLayerC, 0);

        for (int r=0; r<5; r++) {
            for (int c=0; c<5; c++) {
                EXPECT_NEAR( layer->filters[0]->errorMap[r][c], expectedD[r][c], 1e-8 );
            }
        }
    }


    class GetActivationsFixture : public ::testing::Test {
    public:
        virtual void SetUp() {
            Network::deleteNetwork();
            Network::newNetwork();
            Network::getInstance(0)->weightInitFn = &NetMath::uniform;
            fcLayer1 = new FCLayer(0, 9);
            fcLayer2 = new FCLayer(0, 64);
            fcLayer2->prevLayer = fcLayer1;

            convLayer = new ConvLayer(0, 2);
            convLayer->outMapSize = 3;
            convLayer->channels = 1;
            convLayer->filterSize = 3;
            convLayer->assignPrev(fcLayer1);

            fcLayer1->init(0);
            fcLayer2->init(1);

            fcLayer1->actvns = {};
            for (int n=0; n<9; n++) {
                fcLayer1->actvns.push_back(n+1);
            }
            fcLayer2->actvns = {};
            for (int n=0; n<fcLayer2->neurons.size(); n++) {
                fcLayer2->actvns.push_back(n+1);
            }

            convLayer->filters[0]->activationMap = {{1,2,3},{4,5,6},{7,8,9}};
            convLayer->filters[1]->activationMap = {{4,5,6},{7,8,9},{1,2,3}};
        }

        virtual void TearDown() {
            Network::deleteNetwork();
        }

        FCLayer* fcLayer1;
        FCLayer* fcLayer2;
        ConvLayer* convLayer;
    };


    // Returns all activation values from an FC layer
    TEST_F(GetActivationsFixture, getActivations_1) {
        std::vector<double> expected = {1,2,3,4,5,6,7,8,9};
        EXPECT_EQ( NetUtil::getActivations(fcLayer1), expected );
    }

    // Returns all activation values from a ConvLayer
    TEST_F(GetActivationsFixture, getActivations_2) {
        std::vector<double> expected = {1,2,3,4,5,6,7,8,9,4,5,6,7,8,9,1,2,3};
        EXPECT_EQ( NetUtil::getActivations(convLayer), expected );
    }

    // Returns the FCLayer neuron activations in a square (map) subset of the neurons, indicated by the map index and map size
    TEST_F(GetActivationsFixture, getActivations_3) {
        std::vector<double> expected1 = {1,2,3,4};
        std::vector<double> expected2 = {19,20,21,22,23,24,25,26,27};
        std::vector<double> expected3 = {17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};

        EXPECT_EQ( NetUtil::getActivations(fcLayer2, 0, 4), expected1 );
        EXPECT_EQ( NetUtil::getActivations(fcLayer2, 2, 9), expected2 );
        EXPECT_EQ( NetUtil::getActivations(fcLayer2, 1, 16), expected3 );
    }

    // Returns the activations from a Filter in a ConvLayer if map index is provided as second parameter
    TEST_F(GetActivationsFixture, getActivations_4) {
        std::vector<double> expected1 = {4,5,6,7,8,9,1,2,3};
        std::vector<double> expected2 = {1,2,3,4,5,6,7,8,9};

        EXPECT_EQ( NetUtil::getActivations(convLayer, 1, 0), expected1 );
        EXPECT_EQ( NetUtil::getActivations(convLayer, 0, 0), expected2 );
    }

    // Returns all the activations from a PoolLayer correctly
    TEST_F(GetActivationsFixture, getActivations_5) {
        PoolLayer* poolLayer = new PoolLayer(0, 2);
        poolLayer->activations = {{
            {2,5,8,11,14,17},
            {3,6,9,12,15,18},
            {3,6,9,12,15,18},
            {2,5,8,11,14,17},
            {3,6,9,12,15,18},
            {3,6,9,12,15,18}
        }};

        std::vector<double> expected = {2,5,8,11,14,17,3,6,9,12,15,18,3,6,9,12,15,18,2,5,8,11,14,17,3,6,9,12,15,18,3,6,9,12,15,18};

        EXPECT_EQ( NetUtil::getActivations(poolLayer), expected );
    }

    // Returns just one activation map from a PoolLayer when called with a map index parameter
    TEST_F(GetActivationsFixture, getActivations_6) {
        PoolLayer* poolLayer = new PoolLayer(0, 2);
        poolLayer->activations = {
            {{1,2},{3,4}},
            {{5,6},{7,8}}
        };

        std::vector<double> expected1 = {1,2,3,4};
        std::vector<double> expected2 = {5,6,7,8};

        EXPECT_EQ( NetUtil::getActivations(poolLayer, 0, 0), expected1 );
        EXPECT_EQ( NetUtil::getActivations(poolLayer, 1, 0), expected2 );
    }

    TEST(NetUtil, arrayToMap) {
        std::vector<double> testArray = {1,2,3,4,5,6,7,8,9};
        std::vector<std::vector<double> > expected = {{1,2,3},{4,5,6},{7,8,9}};

        EXPECT_EQ( NetUtil::arrayToMap(testArray, 3), expected );
    }

    // Converts the array correctly (Example 1)
    TEST(NetUtil, arrayToVolume_1) {
        std::vector<double> testData = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36};
        std::vector<std::vector<std::vector<double> > > expected = {{{1,2},{3,4}}, {{5,6},{7,8}}, {{9,10},{11,12}}, {{13,14},{15,16}},
              {{17,18},{19,20}}, {{21,22},{23,24}}, {{25,26},{27,28}}, {{29,30},{31,32}}, {{33,34},{35,36}}};

        EXPECT_EQ( NetUtil::arrayToVolume(testData, 9), expected );
    }

    // Converts the array correctly (Example 2)
    TEST(NetUtil, arrayToVolume_2) {
        std::vector<double> testData = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36};
        std::vector<std::vector<std::vector<double> > > expected = { {{1,2,3},{4,5,6},{7,8,9}}, {{10,11,12},{13,14,15},{16,17,18}},
                       {{19,20,21},{22,23,24},{25,26,27}}, {{28,29,30},{31,32,33},{34,35,36}} };
        EXPECT_EQ( NetUtil::arrayToVolume(testData, 4), expected );
    }

    // Converts the array correctly (Example 3)
    TEST(NetUtil, arrayToVolume_3) {
        std::vector<double> testData = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36};
        std::vector<std::vector<std::vector<double> > > expected = {{{1,2,3,4,5,6},{7,8,9,10,11,12},{13,14,15,16,17,18},{19,20,21,22,23,24},
                                    {25,26,27,28,29,30},{31,32,33,34,35,36}}};
        EXPECT_EQ( NetUtil::arrayToVolume(testData, 1), expected );
    }
}


int main (int argc, char** argv) {
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}